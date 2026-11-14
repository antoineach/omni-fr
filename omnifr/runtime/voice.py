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

import queue
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

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
        if voice_ref:
            self.emb = Path(voice_ref).with_suffix(".emb.npy")
            if not self.emb.exists():
                out = Path(tempfile.mkstemp(suffix=".wav")[1])
                self._run(["Bonjour.", str(out), "--clone", str(voice_ref), "--save-speaker-embedding", str(self.emb)])

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
        for k, v in MOODS.get(emotion, {}).items():
            args += ["--control", f"{k}={v}"]
        self._run(args)
        x, sr = sf.read(str(out), dtype="float32", always_2d=True)
        out.unlink(missing_ok=True)
        x = x.mean(1)
        if np.sqrt(np.mean(x ** 2)) < 1e-3 and seed < 4:     # sortie muette observée sans clonage : on réessaie
            return self.synth(text, emotion, seed + 1)
        return x, sr


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


class Speaker:
    """Synthétise phrase par phrase en tâche de fond pour commencer à parler plus tôt."""

    def __init__(self, tts: Zonos3):
        self.tts = tts

    def say(self, text: str, emotion: str, mic: Mic | None = None, vad: Vad | None = None,
            barge_in: bool = False) -> bool:
        import re
        sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()] or [text]
        q: queue.Queue = queue.Queue(maxsize=2)

        def worker():
            for s in sentences:
                q.put(self.tts.synth(s, emotion))
            q.put(None)
        threading.Thread(target=worker, daemon=True).start()
        while (item := q.get()) is not None:
            if not play(*item, mic=mic, vad=vad, barge_in=barge_in):
                return False
        return True
