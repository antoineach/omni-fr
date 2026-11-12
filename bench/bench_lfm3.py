"""Banc temps réel de LFM3-Audio-2B (Liquid AI) dans llama.cpp, sans PyTorch.

Mesure sur la machine courante, CPU seul par défaut (le Pi 5 n'a pas de GPU exploitable) :
1. Synthèse vocale française (``llama-tts``) : facteur temps réel = durée de calcul / durée audio.
2. Transcription française de l'audio produit (``llama cli`` / ``llama-mtmd-cli``) : facteur
   temps réel et texte reconnu (contrôle aller-retour).
3. ``llama-bench`` sur le modèle principal : débit de décodage en jetons/s.
4. Étalonnage : ``llama-bench`` sur Gemma 3 1B Q4_K_M, dont on connaît ~11,5 jetons/s sur Pi 5
   (banc publié, Ollama). Le rapport donne un facteur pour extrapoler au Pi 5.

    python bench/bench_lfm3.py --bin llama.cpp/build/bin --threads 4 --ngl 0 --json lfm3.json

Les modèles sont téléchargés depuis Hugging Face dans ``--models``.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

LFM3_REPO = "LiquidAI/LFM3-Audio-2B-GGUF"
CALIB_REPO = "ggml-org/gemma-3-1b-it-GGUF"
PI5_GEMMA3_1B_TG = 11.53   # jetons/s, Ollama Q4_K_M, documentation Slips (Stratosphere)

TEXTS_FR = [
    "Bonjour ! J'ai allumé la lumière de la cuisine et fermé le volet du salon.",
    "Demain à Bordeaux, il y aura des averses le matin, puis des éclaircies l'après-midi, "
    "avec des températures comprises entre huit et quatorze degrés.",
]


def find_bin(bindir: Path, *names: str) -> list[str] | None:
    for n in names:
        parts = n.split()
        p = bindir / parts[0]
        if p.exists():
            return [str(p)] + parts[1:]
    return None


def download(repo: str, patterns: list[str], dest: Path) -> list[Path]:
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(repo)
    print(f"{repo} : {len(files)} fichiers ; gguf = {[f for f in files if f.endswith('.gguf')]}")
    out = []
    for pat in patterns:
        match = [f for f in files if re.fullmatch(pat, f)]
        if not match:
            sys.exit(f"aucun fichier ne correspond à {pat} dans {repo}")
        out.append(Path(hf_hub_download(repo, match[0], local_dir=dest)))
    return out


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def run(cmd: list[str], timeout: int = 1800) -> tuple[float, str]:
    print("$", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    dt = time.perf_counter() - t0
    out = p.stdout + "\n" + p.stderr
    for line in out.splitlines():   # lignes de performance de llama.cpp, pour le détail par étage
        if re.search(r"(perf|eval time|load time|total time|tokens per second|t/s|codec|vocoder|audio)", line, re.I):
            print("   |", line.strip()[:220])
    if p.returncode != 0:
        print(out[-3000:])
        raise RuntimeError(f"échec ({p.returncode}) : {' '.join(cmd[:2])}")
    return dt, out


def bench_tg(bench: list[str], model: Path, threads: int, ngl: int) -> float:
    _, out = run(bench + ["-m", str(model), "-t", str(threads), "-ngl", str(ngl),
                          "-p", "0", "-n", "64", "-r", "3", "-o", "json"])
    data = json.loads(out[out.index("["):out.rindex("]") + 1])
    return float(data[0]["avg_ts"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="dossier des binaires llama.cpp")
    ap.add_argument("--models", default="models")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ngl", type=int, default=0, help="0 = CPU seul")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    bindir, dest = Path(a.bin), Path(a.models)
    dest.mkdir(parents=True, exist_ok=True)
    tts = find_bin(bindir, "llama-tts")
    asr = find_bin(bindir, "llama cli", "llama-mtmd-cli", "llama-cli")
    bench = find_bin(bindir, "llama-bench")
    print("binaires :", tts, asr, bench)

    # Projecteur audio (encodeur + codec) en Q8_0, comme le recommande Liquid.
    model, mmproj = download(LFM3_REPO, [rf"LFM3-Audio-2B-{a.quant}\.gguf", r"mmproj-LFM3-Audio-2B-Q8_0\.gguf"], dest)
    common = ["-t", str(a.threads), "-ngl", str(a.ngl)]
    res: dict = {"machine": platform.machine(), "systeme": platform.platform(), "threads": a.threads,
                 "ngl": a.ngl, "quant": a.quant, "modele": model.name, "mmproj": mmproj.name, "erreurs": []}

    def section(name, fn):
        try:
            res[name] = fn()
            print(f"[{name}]", json.dumps(res[name], ensure_ascii=False)[:1500], flush=True)
        except Exception as e:  # une étape en échec n'empêche pas les autres
            res["erreurs"].append(f"{name}: {e}")
            print(f"[{name}] ÉCHEC : {e}", flush=True)

    # 1. Débit de décodage du modèle principal et étalonnage Pi 5 (rapides, en premier)
    section("decodage_jetons_s", lambda: round(bench_tg(bench, model, a.threads, a.ngl), 2))

    def calib():
        (c,) = download(CALIB_REPO, [r"gemma-3-1b-it-Q4_K_M\.gguf"], dest)
        tg = bench_tg(bench, c, a.threads, a.ngl)
        return {"gemma3_1b_q4km_jetons_s": round(tg, 2), "pi5_publie": PI5_GEMMA3_1B_TG,
                "facteur_machine_sur_pi5": round(tg / PI5_GEMMA3_1B_TG, 2)}
    section("etalonnage", calib)

    # 2. Synthèse française. Une phrase très courte mesure le coût fixe (chargement, préchauffage),
    #    retiré ensuite pour obtenir le coût de génération par seconde d'audio.
    def synth(text: str, wav: Path) -> tuple[float, float]:
        base = tts + ["-m", str(model), "-mm", str(mmproj), "-p", text, "-o", str(wav)] + common
        try:
            dt, _ = run(base + ["--tts-lang", "fr"])
        except RuntimeError:
            dt, _ = run(base)
        return dt, wav_seconds(wav)

    def tts_all():
        dt0, dur0 = synth("Oui.", dest / "tts_fr_court.wav")
        rows = [{"texte": "Oui.", "audio_s": round(dur0, 2), "calcul_s": round(dt0, 2)}]
        for i, text in enumerate(TEXTS_FR):
            dt, dur = synth(text, dest / f"tts_fr_{i}.wav")
            rows.append({"texte": text, "audio_s": round(dur, 2), "calcul_s": round(dt, 2),
                         "facteur_temps_reel_brut": round(dt / dur, 3),
                         "facteur_temps_reel_net": round((dt - dt0) / max(0.1, dur - dur0), 3)})
        return rows
    section("tts_fr", tts_all)

    # 3. Transcription française de l'audio produit (aller-retour)
    def asr_all():
        rows = []
        for i, text in enumerate(TEXTS_FR):
            src = dest / f"tts_fr_{i}.wav"
            wav16 = dest / f"tts_fr_{i}_16k.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", "16000",
                            "-ac", "1", str(wav16)], check=True)
            dt, out = run(asr + ["-m", str(model), "-mm", str(mmproj), "--audio", str(wav16),
                                 "-sys", "Perform ASR in French.", "-p", "<__media__>", "-n", "128",
                                 "--temp", "0"] + common, timeout=900)
            dur = wav_seconds(wav16)
            rows.append({"reference": text, "sortie": out.strip()[-500:], "audio_s": round(dur, 2),
                         "calcul_s": round(dt, 2), "facteur_temps_reel_brut": round(dt / dur, 3)})
        return rows
    section("asr_fr", asr_all)

    f = res.get("etalonnage", {}).get("facteur_machine_sur_pi5")
    if f and "tts_fr" in res:
        net = max(r["facteur_temps_reel_net"] for r in res["tts_fr"] if "facteur_temps_reel_net" in r)
        res["estimation_pi5"] = {
            "tts_facteur_temps_reel_net": round(net * f, 2),
            "note": "extrapolation linéaire par le débit de décodage ; < 1 = plus rapide que le temps réel",
        }
    print(json.dumps({k: v for k, v in res.items() if k != "asr_fr"}, ensure_ascii=False, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
