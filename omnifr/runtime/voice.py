"""Chaîne audio : micro, mot d'éveil, détection de parole, transcription, voix, lecture interruptible.

- Mot d'éveil : openWakeWord « hey_jarvis » (ONNX, CPU, quelques % d'un cœur).
- Détection de parole : Silero VAD via sherpa-onnx.
- Transcription : Parakeet TDT 0.6B v3 int8 via sherpa-onnx (CPU : une minute en ~2,5 s).
- Voix : ZONOS3-Mini, moteur ggml natif (`zonos3-cli --gpu`), émotion par axes, voix clonée
  à partir d'une empreinte calculée une seule fois.
- Interruption : pendant la lecture, si la VAD entend parler plus de 0,3 s, la lecture s'arrête.
  Sans annulation d'écho (micro XVF3800 ou casque), l'assistant s'entendrait lui-même : l'option
  est donc désactivée par défaut.
"""

from __future__ import annotations

import os
import queue
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

import atexit

_CHILDREN: list = []


def _reap() -> None:
    for p in _CHILDREN:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()


atexit.register(_reap)

SR = 16000
FRAME = 1280  # 80 ms, taille attendue par openWakeWord
PARAKEET_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
                "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2")
SILERO_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

# Émotion du LLM -> doses sur les axes de ZONOS3 (valence, activation, dominance, tempo).
MOODS = {"calme": {"mood.valence": 0.5, "mood.arousal": -1.5, "mood.tempo": -0.5},
         "joie": {"mood.valence": 1.5, "mood.arousal": 1.5},
         "inquietude": {"mood.valence": -1.5, "mood.arousal": 1.0, "mood.dominance": -1.0},
         "ironie": {"mood.valence": -0.5, "mood.dominance": 1.5, "mood.tempo": -0.5},
         "neutre": {}}


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"téléchargement {url}", flush=True)
        urllib.request.urlretrieve(url, dest)
    return dest


class Parakeet:
    def __init__(self, models: Path, threads: int = 4, provider: str = "cpu"):
        import sherpa_onnx
        d = models / "parakeet"
        if not any(d.glob("**/tokens.txt")):
            tmp = fetch(PARAKEET_URL, d / "m.tar.bz2")
            with tarfile.open(tmp) as t:
                t.extractall(d, filter="data")
            tmp.unlink()
        root = next(d.glob("**/tokens.txt")).parent
        f = lambda pat: str(next(root.glob(pat)))  # noqa: E731
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f("encoder*.onnx"), decoder=f("decoder*.onnx"), joiner=f("joiner*.onnx"),
            tokens=f("tokens.txt"), model_type="nemo_transducer", num_threads=threads, provider=provider)

    def __call__(self, x: np.ndarray) -> str:
        s = self.rec.create_stream()
        s.accept_waveform(SR, x.astype(np.float32))
        self.rec.decode_stream(s)
        return s.result.text.strip()


class Vad:
    def __init__(self, models: Path, threshold: float = 0.5):
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(fetch(SILERO_URL, models / "silero_vad.onnx"))
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = 0.25
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.sample_rate = SR
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
        self.window = cfg.silero_vad.window_size
        self.buf = np.zeros(0, np.float32)

    def speaking(self, x: np.ndarray) -> bool:
        self.buf = np.concatenate([self.buf, x])
        while len(self.buf) >= self.window:
            self.vad.accept_waveform(self.buf[: self.window])
            self.buf = self.buf[self.window:]
        return self.vad.is_speech_detected()

    def reset(self) -> None:
        self.vad.reset()
        self.buf = np.zeros(0, np.float32)


class WakeWord:
    def __init__(self, name: str = "hey_jarvis", threshold: float = 0.5):
        import openwakeword
        from openwakeword.model import Model
        openwakeword.utils.download_models([f"{name}_v0.1"])
        self.m = Model(wakeword_models=[name], inference_framework="onnx")
        self.threshold = threshold

    def __call__(self, frame: np.ndarray) -> bool:
        scores = self.m.predict((np.clip(frame, -1, 1) * 32767).astype(np.int16))
        return max(scores.values()) >= self.threshold

    def reset(self) -> None:
        self.m.reset()


class Zonos3:
    """ZONOS3-Mini via son CLI ggml. Une empreinte de voix est calculée une fois, puis réutilisée."""

    def __init__(self, root: Path, voice_ref: Path | None = None, gpu: bool = True):
        self.root = root
        self.cli = str(next(root.glob("servers/ggml/build*/**/zonos3-cli")))
        self.model = str(root / "servers/ggml/gguf/backbone-ptq.gguf")
        self.dev = "--gpu" if gpu else "--cpu"
        self.emb: Path | None = None
        self.clone_ref: Path | None = None
        if voice_ref:
            emb = Path(voice_ref).with_suffix(".emb.npy")
            try:
                if not emb.exists():
                    out = Path(tempfile.mkstemp(suffix=".wav")[1])
                    self._run(["Bonjour.", str(out), "--clone", str(voice_ref), "--save-speaker-embedding", str(emb)])
                self.emb = emb
            except RuntimeError as e:
                # repli : clonage recalculé à chaque phrase (plus lent, mais vérifié au banc)
                print(f"empreinte de voix non enregistrée ({str(e)[:120]}) ; clonage à chaque phrase", flush=True)
                self.clone_ref = Path(voice_ref)

    def _run(self, args: list[str]) -> None:
        r = subprocess.run([self.cli, self.model, "--tts"] + args + [self.dev], capture_output=True, text=True,
                           cwd=self.root / "servers/ggml")
        if r.returncode != 0:
            raise RuntimeError("zonos3-cli : " + "\n".join(
                x for x in (r.stdout + r.stderr).splitlines()[-15:] if "repack" not in x))

    def synth(self, text: str, emotion: str = "neutre", seed: int = 1) -> tuple[np.ndarray, int]:
        import soundfile as sf
        out = Path(tempfile.mkstemp(suffix=".wav")[1])
        args = [text, str(out), "--seed", str(seed)]
        if self.emb:
            args += ["--speaker-embedding", str(self.emb)]
        elif self.clone_ref:
            args += ["--clone", str(self.clone_ref)]
        for k, v in MOODS.get(emotion, {}).items():
            args += ["--control", f"{k}={v}"]
        self._run(args)
        x, sr = sf.read(str(out), dtype="float32", always_2d=True)
        out.unlink(missing_ok=True)
        x = x.mean(1)
        if np.sqrt(np.mean(x ** 2)) < 1e-3 and seed < 4:     # sortie muette observée sans clonage : on réessaie
            return self.synth(text, emotion, seed + 1)
        return x, sr


class Zonos3Server:
    """ZONOS3-Mini en serveur : le modèle reste chargé (le CLI le recharge à chaque phrase).

    La voix est fixée par un extrait de référence et une graine constante : sans cela ZONOS3 tire
    une voix différente à chaque requête.
    """

    def __init__(self, root: Path, voice_ref: Path | None = None, gpu: bool = True, port: int = 1919,
                 log: Path | None = None):
        import base64
        self.root, self.port = root, port
        exe = str(next(root.glob("servers/ggml/build*/**/zonos3-server")))
        cmd = [exe, str(root / "servers/ggml/gguf/backbone-ptq.gguf"), "--port", str(port),
               "--gpu" if gpu else "--cpu"]
        dv = root / "assets/default_voices"
        if dv.exists():
            cmd += ["--tts-default-voices-dir", str(dv)]
        self.proc = subprocess.Popen(cmd, cwd=root / "servers/ggml", stdout=open(log or os.devnull, "w"),
                                     stderr=subprocess.STDOUT)
        _CHILDREN.append(self.proc)
        self.ref = base64.b64encode(Path(voice_ref).read_bytes()).decode() if voice_ref else None
        t0 = time.time()
        while time.time() - t0 < 180:
            if self.proc.poll() is not None:
                raise RuntimeError("zonos3-server s'est arrêté au démarrage")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v2/capabilities", timeout=3)
                return
            except Exception:
                time.sleep(0.5)
        raise TimeoutError("zonos3-server non prêt")

    def synth(self, text: str, emotion: str = "neutre", seed: int = 7) -> tuple[np.ndarray, int]:
        import io
        import json as _json
        import soundfile as sf
        body = {"text": text, "format": "wav", "seed": seed, "controls": MOODS.get(emotion, {})}
        if self.ref:
            body["speaker_audio_base64"] = self.ref
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v2/generate", data=_json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                x, sr = sf.read(io.BytesIO(r.read()), dtype="float32", always_2d=True)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"zonos3-server {e.code} : {e.read().decode('utf-8', 'replace')[:400]}") from None
        return x.mean(1), sr

    def close(self) -> None:
        self.proc.terminate()


# Émotion -> instruction de jeu pour Qwen3-TTS CustomVoice (l'anglais est la langue d'instruction la mieux suivie).
QWEN_INSTRUCT = {"calme": "Speak in a calm, soft and reassuring tone.",
                 "joie": "Speak in a cheerful, happy and lively tone.",
                 "inquietude": "Speak in a worried, urgent and tense tone.",
                 "ironie": "Speak in a dry, ironic, slightly amused tone, like a witty British butler.",
                 "neutre": ""}


class Qwen3TTS:
    """Client du serveur Qwen3-TTS (qwen_tts_server.py), lancé dans .venv-voix.

    qwen-tts exige transformers==4.57.3 : il vit dans son propre environnement Python, le modèle
    reste chargé sur le GPU, et l'assistant lui envoie les phrases en HTTP local.
    """

    def __init__(self, mode: str = "expressif", size: str = "0.6B", voice_ref: Path | None = None,
                 speaker: str = "", asr=None, port: int = 8095, log: Path | None = None):
        import sys
        repo = Path(__file__).resolve().parents[2]
        py = repo / ".venv-voix" / "bin" / "python"
        if not py.exists():
            raise FileNotFoundError(f"{py} absent : lancer « bash scripts/installer_voix.sh »")
        cmd = [str(py), str(Path(__file__).with_name("qwen_tts_server.py")), "--port", str(port), "--mode", mode,
               "--taille", size]
        if mode == "clone":
            if not voice_ref:
                raise ValueError("mode clone : donner --voix extrait.wav")
            cmd += ["--voix", str(voice_ref)]
            if asr is not None:      # la transcription de l'extrait améliore le clonage
                import soundfile as sf
                from bench.eval_audio import resample
                x, sr = sf.read(str(voice_ref), dtype="float32", always_2d=True)
                cmd += ["--texte-voix", asr(resample(x.mean(1), sr))]
        elif speaker:
            cmd += ["--nom", speaker]
        self.port = port
        self.log = log or Path(os.devnull)
        self.proc = subprocess.Popen(cmd, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        _CHILDREN.append(self.proc)
        t0 = time.time()
        while time.time() - t0 < 600:
            if self.proc.poll() is not None:
                lines = Path(self.log).read_text(errors="replace").splitlines() if self.log.exists() else []
                tail = "\n".join(lines[-25:])
                raise RuntimeError(f"serveur Qwen3-TTS arrêté :\n{tail}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            raise TimeoutError("serveur Qwen3-TTS non prêt")
        if self.log.exists():
            for line in self.log.read_text(errors="replace").splitlines():
                if line.startswith("voix disponibles"):
                    print(line, flush=True)

    def stream(self, text: str, emotion: str = "neutre"):
        """Génère en flux : renvoie (sr, itérateur de morceaux float32) dès le premier morceau."""
        import json as _json
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/tts",
                                     data=_json.dumps({"text": text, "emotion": emotion}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Qwen3-TTS : {e.read().decode('utf-8', 'replace')[:400]}") from None
        sr = int(r.headers["X-Sample-Rate"])

        def chunks():
            buf = b""
            with r:
                while True:
                    data = r.read1(65536) if hasattr(r, "read1") else r.read(4096)
                    if not data:
                        break
                    buf += data
                    n = len(buf) // 4 * 4
                    if n:
                        yield np.frombuffer(buf[:n], dtype=np.float32)
                        buf = buf[n:]
        return sr, chunks()

    def synth(self, text: str, emotion: str = "neutre") -> tuple[np.ndarray, int]:
        sr, it = self.stream(text, emotion)
        parts = list(it)
        return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), sr

    def close(self) -> None:
        self.proc.terminate()


def make_tts(root: Path, voice_ref: Path | None, gpu: bool, log: Path | None = None, engine: str = "qwen",
             speaker: str = "", size: str = "0.6B", asr=None):
    """Qwen3-TTS par défaut (expressif, ou clone si --voix-mode clone) ; ZONOS3 en repli."""
    if engine.startswith("qwen"):
        mode = "clone" if engine == "qwen-clone" else "expressif"
        try:
            return Qwen3TTS(mode, size, voice_ref, speaker, asr=asr,
                            log=(log.parent / "qwen-tts.log") if log else None)
        except Exception as e:
            print(f"Qwen3-TTS indisponible : {type(e).__name__}: {e}\n(journal complet : {log.parent / 'qwen-tts.log' if log else '?'})"
                  " ; repli sur ZONOS3", flush=True)
    try:
        return Zonos3Server(root, voice_ref, gpu, log=log)
    except Exception as e:
        print(f"zonos3-server indisponible ({e}) ; repli sur zonos3-cli (plus lent)", flush=True)
        return Zonos3(root, voice_ref, gpu)


class Mic:
    """Flux micro 16 kHz mono, en blocs de 80 ms, dans une file."""

    def __init__(self, device=None):
        import sounddevice as sd
        self.q: queue.Queue = queue.Queue()
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=FRAME,
                                     device=device, callback=lambda d, *_: self.q.put(d[:, 0].copy()))
        self.stream.start()

    def read(self, timeout: float | None = None) -> np.ndarray:
        return self.q.get(timeout=timeout)

    def drain(self) -> None:
        while not self.q.empty():
            self.q.get_nowait()


def record_utterance(mic: Mic, vad: Vad, max_s: float = 60.0, end_silence: float = 0.8,
                     start_timeout: float = 6.0) -> np.ndarray | None:
    """Enregistre jusqu'à ``end_silence`` s de silence après la parole (une minute au plus)."""
    vad.reset()
    chunks, started, silent, t0 = [], False, 0.0, time.time()
    while True:
        x = mic.read()
        chunks.append(x)
        sp = vad.speaking(x)
        dt = len(x) / SR
        if sp:
            started, silent = True, 0.0
        elif started:
            silent += dt
            if silent >= end_silence:
                break
        elif time.time() - t0 > start_timeout:
            return None
        if time.time() - t0 > max_s:
            break
    return np.concatenate(chunks)


def play(x: np.ndarray, sr: int, mic: Mic | None = None, vad: Vad | None = None, barge_in: bool = False) -> bool:
    """Joue l'audio. Renvoie False si l'utilisateur a coupé la parole."""
    import sounddevice as sd
    sd.play(x, sr)
    if not (barge_in and mic and vad):
        sd.wait()
        return True
    mic.drain()
    vad.reset()
    speech, end = 0.0, time.time() + len(x) / sr
    while time.time() < end:
        try:
            frame = mic.read(timeout=0.2)
        except queue.Empty:
            continue
        speech = speech + len(frame) / SR if vad.speaking(frame) else 0.0
        if speech >= 0.3:
            sd.stop()
            return False
    sd.wait()
    return True


def _play_stream(sr: int, chunks, mic=None, vad=None, barge_in: bool = False, t0: float | None = None,
                 verbose: bool = True, text: str = "") -> bool:
    """Joue les morceaux au fil de leur arrivée, avec un tampon adaptatif.

    Si la génération va à peine plus vite que la parole (cas de la 2070, facteur ~1), lire chaque
    morceau dès son arrivée vide la carte son : craquements (« underrun »). On mesure donc la
    vitesse de génération sur les premiers morceaux et on attend juste assez d'audio d'avance pour
    finir la phrase sans trou. Renvoie False si l'utilisateur a coupé la parole.
    """
    import queue as _q
    import sounddevice as sd
    est_total = max(1.0, 0.075 * len(text))            # ~13 caractères par seconde parlée
    buf: _q.Queue = _q.Queue()
    stats = {"gen": 0, "done": False, "t_first": None}

    def producer():
        try:
            for c in chunks:
                if stats["t_first"] is None:
                    stats["t_first"] = time.time()
                stats["gen"] += len(c)
                buf.put(c)
        finally:
            stats["done"] = True
            buf.put(None)
    threading.Thread(target=producer, daemon=True).start()

    first = buf.get()
    if first is None:
        return True
    if verbose and t0 is not None:
        print(f"   [voix : premier son généré après {time.time() - t0:.2f} s]", flush=True)
    pending = [first]
    first_s = len(first) / sr
    # tampon : attendre que (audio d'avance) >= (reste à générer) x (1 - vitesse relative)
    while not stats["done"]:
        el = max(1e-3, time.time() - stats["t_first"])
        gen_s = stats["gen"] / sr
        # vitesse mesurée sur au moins 0,8 s après le premier morceau. Un morceau du serveur arrive
        # souvent découpé en plusieurs lectures réseau simultanées : compter les morceaux ne suffit pas.
        speed = (gen_s - first_s) / el if el >= 0.8 else None
        if speed is None:
            need = float("inf")
        else:
            need = 0.3 if speed >= 1.3 else max(0.3, (est_total - gen_s) * (1 - speed) / max(speed, 0.3) + 0.4)
        if gen_s >= need:
            break
        try:
            c = buf.get(timeout=0.05)
        except _q.Empty:
            continue
        if c is None:
            break
        pending.append(c)
    if verbose and t0 is not None:
        print(f"   [voix : lecture après {time.time() - t0:.2f} s, {stats['gen'] / sr:.1f} s d'avance]", flush=True)

    if barge_in and mic and vad:
        mic.drain()
        vad.reset()
    speech, total = 0.0, 0
    with sd.OutputStream(samplerate=sr, channels=1, dtype="float32", latency="high") as out:
        while True:
            c = pending.pop(0) if pending else buf.get()
            if c is None:
                break
            out.write(c.reshape(-1, 1))
            total += len(c)
            if barge_in and mic and vad:
                while not mic.q.empty():
                    f = mic.q.get_nowait()
                    speech = speech + len(f) / SR if vad.speaking(f) else 0.0
                if speech >= 0.3:
                    if hasattr(chunks, "close"):
                        try:
                            chunks.close()
                        except Exception:
                            pass
                    return False
    if verbose and t0 is not None:
        print(f"   [voix : {total / sr:.1f} s d'audio, fini {time.time() - t0:.1f} s après la demande]", flush=True)
    return True


class Speaker:
    """Synthétise phrase par phrase en tâche de fond pour commencer à parler plus tôt."""

    def __init__(self, tts, verbose: bool = True):
        self.tts, self.verbose = tts, verbose

    def _say_stream(self, text, emotion, mic, vad, barge_in) -> bool:
        t0 = time.time()
        try:
            sr, it = self.tts.stream(text, emotion)
        except Exception as e:
            print(f"   ⚠ voix : {e}", flush=True)
            return True
        return _play_stream(sr, it, mic, vad, barge_in, t0=t0, verbose=self.verbose, text=text)

    def say(self, text: str, emotion: str, mic: Mic | None = None, vad: Vad | None = None,
            barge_in: bool = False) -> bool:
        import re
        if hasattr(self.tts, "stream"):
            return self._say_stream(re.sub(r"(\d)\.(\d)", r"\1,\2", text), emotion, mic, vad, barge_in)
        text = re.sub(r"(\d)\.(\d)", r"\1,\2", text)      # 15.4 -> 15,4 : lu « quinze virgule quatre »
        sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()] or [text]
        q: queue.Queue = queue.Queue(maxsize=2)

        def worker():
            try:
                for s in sentences:
                    t0 = time.time()
                    x, sr = self.tts.synth(s, emotion)
                    if self.verbose:
                        print(f"   [voix : {len(x) / sr:.1f} s d'audio en {time.time() - t0:.1f} s]", flush=True)
                    q.put((x, sr))
            except Exception as e:
                print(f"   ⚠ voix : {e}", flush=True)
            q.put(None)
        threading.Thread(target=worker, daemon=True).start()
        while (item := q.get()) is not None:
            if not play(*item, mic=mic, vad=vad, barge_in=barge_in):
                return False
        return True
