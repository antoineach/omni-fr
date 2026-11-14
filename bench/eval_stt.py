"""Banc de transcription française : trois moteurs, voix humaines réelles et phrases domotiques.

Jeux :
  - « humain » : Multilingual LibriSpeech français (lectures LibriVox, vrais locuteurs, micros variés)
  - « domotique » : phrases des scénarios, synthétisées par LFM3-Audio (voix homme/femme)
Conditions : propre, brouhaha 10 dB, brouhaha 0 dB, bruit rose 5 dB.
Moteurs (chacun isolé : un échec n'arrête pas les autres) :
  - parakeet-v3-int8 : NVIDIA parakeet-tdt-0.6b-v3, sherpa-onnx int8, CPU
  - nemotron-3.5-streaming : nvidia/nemotron-3.5-asr-streaming-0.6b, transformers ≥ 5.13 (PyTorch CPU,
    hors cible de déploiement : sert à mesurer la qualité ; le moteur de prod serait NeMo-Speech.cpp/ONNX)
  - lfm3-audio : LFM3-Audio-2B Q4_0, llama-server

WER calculé après normalisation (minuscules, ponctuation, nombres en lettres).

    python bench/eval_stt.py --bin llama.cpp/build/bin --json eval-stt.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.eval_audio import REPO as LFM_REPO, transcribe as lfm_transcribe  # noqa: E402
from bench.eval_llm import wait_ready  # noqa: E402
from bench.voice_common import CONDITIONS, Parakeet, mls_french, noisy, read_wav, resample  # noqa: E402
from omnifr.metrics import normalize_fr, wer  # noqa: E402
from omnifr.sim.scenarios import load  # noqa: E402


class Nemotron:
    name = "nemotron-3.5-streaming"

    def __init__(self, threads: int):
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor
        torch.set_num_threads(threads)
        mid = "nvidia/nemotron-3.5-asr-streaming-0.6b"
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForRNNT.from_pretrained(mid, torch_dtype=torch.float32).eval()
        self.sr = self.proc.feature_extractor.sampling_rate

    def __call__(self, x: np.ndarray) -> str:
        x = resample(x, 16000, self.sr)
        inp = self.proc(x, sampling_rate=self.sr, language="fr")
        with self.torch.inference_mode():
            out = self.model.generate(**inp, return_dict_in_generate=True)
        t = self.proc.decode(out.sequences, skip_special_tokens=True)
        return (t[0] if isinstance(t, list) else t).strip()


class LFM3:
    name = "lfm3-audio"

    def __init__(self, binp: Path, model: Path, mmproj: Path, threads: int, port: int = 8089):
        server = [str(binp / "llama-server")] if (binp / "llama-server").exists() else [str(binp / "llama"), "serve"]
        cmd = server + ["-m", str(model), "-mm", str(mmproj), "--port", str(port), "-c", "4096",
                        "-t", str(threads), "-np", "1"]
        self.port = port
        self.proc = subprocess.Popen(cmd, stdout=open("serveur-stt.log", "w"), stderr=subprocess.STDOUT)
        wait_ready(port, self.proc)

    def __call__(self, x: np.ndarray) -> str:
        return lfm_transcribe(self.port, x)[0]

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def synth_domotique(binp: Path, model: Path, mmproj: Path, n: int, dest: Path, threads: int):
    texts = []
    for s in load(Path(__file__).resolve().parents[1] / "data" / "scenarios_fr_seed.jsonl"):
        t = [d["texte"] for d in s.dialogue if d["role"] == "user"][-1]
        if "[" not in t and t not in texts:
            texts.append(t)
    out = []
    for i, t in enumerate(texts[:n]):
        p = dest / f"dom_{i}.wav"
        cmd = [str(binp / "llama-tts"), "-m", str(model), "-mm", str(mmproj), "-p", t, "-o", str(p),
               "-t", str(threads), "--tts-voice", "male" if i % 2 else "female"]
        r = subprocess.run(cmd + ["--tts-lang", "fr"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)
        x, sr = read_wav(p)
        out.append((t, resample(x, sr)))
    return out


def run_engine(eng, sets: dict, rng_seed: int = 0) -> dict:
    res = {}
    for set_name, items in sets.items():
        refs = [t for t, _ in items]
        clean = [x for _, x in items]
        rng = np.random.default_rng(rng_seed)
        for kind, snr, label in CONDITIONS:
            hyps, secs, durs = [], 0.0, 0.0
            for i in range(len(clean)):
                y = noisy(clean, i, kind, snr, rng)
                t0 = time.perf_counter()
                try:
                    h = eng(y)
                except Exception as e:  # une phrase ratée compte comme vide
                    h = ""
                    print(f"  erreur {type(e).__name__}: {e}", flush=True)
                secs += time.perf_counter() - t0
                durs += len(y) / 16000
                hyps.append(h)
            key = f"{set_name} / {label}"
            res[key] = {"wer": round(wer(refs, hyps), 4), "facteur_temps_reel": round(secs / durs, 3),
                        "exemples": [{"ref": normalize_fr(r), "hyp": normalize_fr(h)}
                                     for r, h in list(zip(refs, hyps))[:3]]}
            print(f"{eng.name:<24} {key:<28} WER {100 * res[key]['wer']:5.1f} %  "
                  f"temps réel x{res[key]['facteur_temps_reel']}", flush=True)
            if label == "propre":
                for ex in res[key]["exemples"]:
                    print(f"    réf : {ex['ref']}\n    hyp : {ex['hyp']}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-humain", type=int, default=24)
    ap.add_argument("--n-domotique", type=int, default=16)
    ap.add_argument("--moteurs", default="parakeet-v3-int8 nemotron-3.5-streaming lfm3-audio")
    ap.add_argument("--models", default="models")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    dest = Path(a.models)
    dest.mkdir(exist_ok=True)
    binp = Path(a.bin)
    model = Path(hf_hub_download(LFM_REPO, "LFM3-Audio-2B-Q4_0.gguf", local_dir=dest))
    mmproj = Path(hf_hub_download(LFM_REPO, "mmproj-LFM3-Audio-2B-Q8_0.gguf", local_dir=dest))

    sets = {}
    try:
        sets["humain"] = mls_french(a.n_humain, dest)
        print(f"humain : {len(sets['humain'])} phrases MLS, "
              f"{sum(len(x) for _, x in sets['humain']) / 16000:.0f} s", flush=True)
    except Exception:
        traceback.print_exc()
    sets["domotique"] = synth_domotique(binp, model, mmproj, a.n_domotique, dest, a.threads)
    print(f"domotique : {len(sets['domotique'])} phrases synthétisées", flush=True)

    results, wanted = {}, a.moteurs.split()
    factories = {
        "parakeet-v3-int8": lambda: Parakeet(dest, a.threads),
        "nemotron-3.5-streaming": lambda: Nemotron(a.threads),
        "lfm3-audio": lambda: LFM3(binp, model, mmproj, a.threads),
    }
    for name in wanted:
        print(f"\n=== {name} ===", flush=True)
        eng = None
        try:
            t0 = time.perf_counter()
            eng = factories[name]()
            print(f"chargé en {time.perf_counter() - t0:.0f} s", flush=True)
            results[name] = run_engine(eng, sets)
        except Exception:
            traceback.print_exc()
            results[name] = {"erreur": traceback.format_exc(limit=3)}
        finally:
            if eng is not None and hasattr(eng, "close"):
                eng.close()

    summary = {m: {k: v["wer"] for k, v in r.items() if isinstance(v, dict)} for m, r in results.items()}
    print("RESUME_STT", json.dumps(summary, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
