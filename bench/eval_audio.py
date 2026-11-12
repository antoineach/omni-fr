"""Transcription française bruitée par LFM3-Audio-2B (llama.cpp), sans PyTorch.

1. Les phrases utilisateur des scénarios sont synthétisées par LFM3-Audio (voix homme et femme).
2. Chaque phrase est mélangée à deux bruits : bruit rose et brouhaha (trois autres phrases
   superposées, comme une pièce où d'autres gens parlent), à 20, 10, 5 et 0 dB de RSB.
3. Le serveur llama.cpp transcrit ; on calcule le WER par condition.

Limite : une voix de synthèse propre est plus facile qu'un humain réel au micro. Ce banc donne
une borne optimiste ; le jeu enregistré chez vous reste la mesure qui compte.

    python bench/eval_audio.py --bin llama.cpp/build/bin --threads 4 --json eval-audio.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.eval_llm import wait_ready  # noqa: E402
from omnifr.metrics import wer  # noqa: E402
from omnifr.sim.scenarios import load  # noqa: E402

REPO = "LiquidAI/LFM3-Audio-2B-GGUF"
SNRS = [None, 20, 10, 5, 0]


def read_wav(p: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(p)) as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    elif sw == 4:
        x = np.frombuffer(raw, np.float32).astype(np.float32)
        if np.abs(x).max() > 10:          # entiers 32 bits
            x = np.frombuffer(raw, np.int32).astype(np.float32) / 2**31
    else:
        raise ValueError(f"largeur d'échantillon non gérée : {sw}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x, sr


def resample(x: np.ndarray, sr: int, to: int = 16000) -> np.ndarray:
    if sr == to:
        return x
    t = np.arange(int(len(x) * to / sr)) * sr / to
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def wav_bytes(x: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    f = np.fft.rfftfreq(n)
    spec = rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    y = np.fft.irfft(spec, n)
    return (y / (np.std(y) + 1e-9)).astype(np.float32)


def mix(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = np.resize(noise, len(clean))
    ps, pn = np.mean(clean ** 2) + 1e-12, np.mean(noise ** 2) + 1e-12
    y = clean + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return y / max(1.0, np.abs(y).max())


def transcribe(port: int, x: np.ndarray) -> tuple[str, float]:
    b64 = base64.b64encode(wav_bytes(x)).decode()
    payload = {"messages": [
        {"role": "system", "content": "Perform ASR in French."},
        {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}],
        "temperature": 0, "max_tokens": 160}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return (out["choices"][0]["message"].get("content") or "").strip(), time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quant", default="Q4_0")
    ap.add_argument("--n", type=int, default=16, help="nombre de phrases")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--models", default="models")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    dest = Path(a.models)
    model = Path(hf_hub_download(REPO, f"LFM3-Audio-2B-{a.quant}.gguf", local_dir=dest))
    mmproj = Path(hf_hub_download(REPO, "mmproj-LFM3-Audio-2B-Q8_0.gguf", local_dir=dest))
    binp = Path(a.bin)

    # Phrases utilisateur : dernier tour utilisateur de chaque scénario, sans les marques [inaudible].
    texts = []
    for s in load(Path(__file__).resolve().parents[1] / "data" / "scenarios_fr_seed.jsonl"):
        t = [d["texte"] for d in s.dialogue if d["role"] == "user"][-1]
        if "[" not in t and t not in texts:
            texts.append(t)
    texts = texts[: a.n]

    clean = []
    for i, t in enumerate(texts):
        out = dest / f"user_{i}.wav"
        voice = "male" if i % 2 else "female"
        cmd = [str(binp / "llama-tts"), "-m", str(model), "-mm", str(mmproj), "-p", t, "-o", str(out),
               "-t", str(a.threads), "--tts-voice", voice]
        p = subprocess.run(cmd + ["--tts-lang", "fr"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if p.returncode != 0:
            subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)
        x, sr = read_wav(out)
        clean.append(resample(x, sr))
        print(f"synthèse {i + 1}/{len(texts)} : {len(clean[-1]) / 16000:.1f} s", flush=True)

    cmd = [str(binp / "llama"), "serve", "-m", str(model), "-mm", str(mmproj), "--port", str(a.port),
           "-c", "4096", "-t", str(a.threads), "-np", "1"]
    log = open("serveur-audio.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    rng = np.random.default_rng(0)
    results = {}
    try:
        wait_ready(a.port, proc)
        for kind in ("rose", "brouhaha"):
            for snr in SNRS:
                if snr is None and kind == "brouhaha":
                    continue
                label = "propre" if snr is None else f"{kind} {snr} dB"
                hyps, secs, durs = [], [], []
                for i, x in enumerate(clean):
                    if snr is None:
                        y = x
                    elif kind == "rose":
                        y = mix(x, pink(len(x), rng), snr)
                    else:
                        others = [clean[(i + k) % len(clean)] for k in (1, 2, 3)]
                        babble = sum(np.resize(o, len(x)) for o in others)
                        y = mix(x, babble, snr)
                    h, dt = transcribe(a.port, y)
                    hyps.append(h)
                    secs.append(dt)
                    durs.append(len(y) / 16000)
                results[label] = {"wer": round(wer(texts, hyps), 4),
                                  "facteur_temps_reel": round(sum(secs) / sum(durs), 3),
                                  "exemples": [{"ref": r, "hyp": h} for r, h in list(zip(texts, hyps))[:4]]}
                print(f"{label:<16} WER {100 * results[label]['wer']:5.1f} %   "
                      f"temps réel x{results[label]['facteur_temps_reel']}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("RESUME_AUDIO", json.dumps({k: v["wer"] for k, v in results.items()}, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps({"phrases": texts, "resultats": results}, ensure_ascii=False, indent=2),
                                encoding="utf-8")


if __name__ == "__main__":
    main()
