"""Banc de voix émotionnelle en français : WAV à écouter, vitesse et intelligibilité.

Moteurs (un par job, dépendances incompatibles entre eux) :
  - chatterbox : Chatterbox Multilingual (Resemble AI), language_id="fr", clonage par extrait de
    référence, intensité réglée par « exaggeration » (pas d'émotion nommée)
  - zonos : ZONOS3-Mini (Zyphra), clonage ; si la bibliothèque expose un vecteur d'émotion
    (comme Zonos v0.1), on le renseigne, sinon on le signale dans le journal

Voix de référence : un extrait MLS français (lecteur LibriVox, enregistrement du domaine public).
C'est un choix de banc d'essai ; la voix finale doit être conçue ou venir d'une personne qui consent.

Intelligibilité : la sortie est retranscrite par parakeet-v3 (sherpa-onnx) et comparée au texte.

    python bench/eval_tts.py --engine chatterbox --out tts-wav --json eval-tts-chatterbox.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.voice_common import Parakeet, load_any, mls_french, resample  # noqa: E402
from omnifr.metrics import normalize_fr, wer  # noqa: E402

PHRASES = [
    ("calme", "Bonsoir. Toutes les lumières du rez-de-chaussée sont éteintes, et la porte d'entrée est verrouillée."),
    ("joie", "Excellente nouvelle ! Votre colis vient d'arriver, et il fait vingt-quatre degrés dehors."),
    ("inquietude", "Attention, je détecte de la fumée dans la cuisine. Voulez-vous que j'appelle les secours ?"),
    ("ironie", "Bien sûr, monsieur. Mettre le chauffage à fond avec les fenêtres ouvertes, quelle idée brillante."),
]
# Intensité Chatterbox par émotion, puis balayage de l'intensité sur une même phrase.
CB_EXAG = {"calme": 0.3, "joie": 0.8, "inquietude": 0.7, "ironie": 0.5}
CB_SWEEP = [0.25, 0.5, 0.75, 1.0]
# Ordre du vecteur d'émotion de Zonos v0.1 : joie, tristesse, dégoût, peur, surprise, colère, autre, neutre.
ZONOS_EMO = {
    "calme": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.6],
    "joie": [0.7, 0.02, 0.02, 0.02, 0.15, 0.02, 0.05, 0.02],
    "inquietude": [0.02, 0.15, 0.02, 0.6, 0.1, 0.02, 0.05, 0.04],
    "ironie": [0.25, 0.02, 0.15, 0.02, 0.1, 0.05, 0.3, 0.11],
}


def save(path: Path, x: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(str(path), np.clip(x, -1, 1), sr)


class Chatterbox:
    def __init__(self, ref: Path):
        import torch
        torch.set_num_threads(4)
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        self.m = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        self.sr, self.ref = self.m.sr, str(ref)

    def jobs(self):
        for emo, t in PHRASES:
            yield f"{emo}", t, {"exaggeration": CB_EXAG[emo]}
        for e in CB_SWEEP:
            yield f"balayage_joie_exag{e}", PHRASES[1][1], {"exaggeration": e}

    def __call__(self, text: str, opts: dict) -> np.ndarray:
        cfg = 0.3 if opts["exaggeration"] >= 0.7 else 0.5   # conseil de l'auteur : cfg plus bas si très expressif
        w = self.m.generate(text, language_id="fr", audio_prompt_path=self.ref,
                            exaggeration=opts["exaggeration"], cfg_weight=cfg)
        return w.squeeze().cpu().numpy().astype(np.float32)


class ChatterboxCpp:
    """Même modèle Chatterbox Multilingual, quantifié en GGUF, exécuté par codec.cpp (llama.cpp + codec ggml)."""

    def __init__(self, ref: Path, binp: Path, dest: Path, quant: str):
        import subprocess
        from huggingface_hub import hf_hub_download
        self.sp, self.bin, self.ref = subprocess, str(binp), str(ref)
        files = [f"chatterbox-mtl-{quant}.gguf", f"codec-{quant}.gguf"]
        for repo in ("hans00/Chatterbox-Multilingual-TTS-GGUF", "BricksDisplay/Chatterbox-Multilingual-TTS-GGUF"):
            try:
                self.backbone, self.codec = (hf_hub_download(repo, f, local_dir=dest) for f in files)
                print(f"GGUF depuis {repo} : {files}", flush=True)
                break
            except Exception as e:
                print(f"{repo} : {e}", flush=True)
        h = subprocess.run([self.bin, "synthesize", "--help"], capture_output=True, text=True)
        self.help = h.stdout + h.stderr
        print(self.help[:3000], flush=True)
        self.exag = next((f for f in ("--exaggeration", "--exag", "--emotion") if f in self.help), None)
        self.sr = 24000
        self.out = dest / "cpp_tmp.wav"

    def jobs(self):
        for emo, t in PHRASES:
            yield f"cpp_{emo}", t, {"exaggeration": CB_EXAG[emo]}
        if self.exag:
            for e in CB_SWEEP:
                yield f"cpp_balayage_joie_exag{e}", PHRASES[1][1], {"exaggeration": e}

    def __call__(self, text: str, opts: dict) -> np.ndarray:
        import unicodedata
        cmd = [self.bin, "synthesize", "--backbone", self.backbone, "--model", self.codec,
               "--text", "[fr]" + unicodedata.normalize("NFKD", text.lower()),
               "--ref-audio", self.ref, "--output", str(self.out)]
        if self.exag:
            cmd += [self.exag, str(opts["exaggeration"])]
        r = self.sp.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-1500:])
        import soundfile as sf
        x, self.sr = sf.read(str(self.out), dtype="float32", always_2d=True)
        return x.mean(1)


class Zonos:
    def __init__(self, ref: Path, repo: str):
        import torch
        import torchaudio
        torch.set_num_threads(4)
        from zonos.conditioning import make_cond_dict
        from zonos.model import Zonos as Z
        self.torch, self.make = torch, make_cond_dict
        self.m = Z.from_pretrained(repo, device="cpu")
        wav, sr = torchaudio.load(str(ref))
        self.spk = self.m.make_speaker_embedding(wav, sr)
        self.sr = self.m.autoencoder.sampling_rate
        params = inspect.signature(make_cond_dict).parameters
        self.has_emotion = "emotion" in params
        print(f"make_cond_dict accepte : {', '.join(params)}", flush=True)
        if not self.has_emotion:
            print("ATTENTION : pas de paramètre « emotion » ; seul le texte porte l'émotion", flush=True)
        self.lang = "fr-fr"

    def jobs(self):
        for emo, t in PHRASES:
            yield emo, t, {"emotion": ZONOS_EMO[emo]}
        yield "joie_sans_emotion", PHRASES[1][1], {}

    def __call__(self, text: str, opts: dict) -> np.ndarray:
        kw = {"text": text, "speaker": self.spk, "language": self.lang}
        if self.has_emotion and "emotion" in opts:
            kw["emotion"] = opts["emotion"]
        with self.torch.inference_mode():
            codes = self.m.generate(self.m.prepare_conditioning(self.make(**kw)))
            w = self.m.autoencoder.decode(codes)[0].cpu().numpy()
        return w.reshape(-1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["chatterbox", "zonos", "chatterbox-cpp"], required=True)
    ap.add_argument("--tts-cli", default="codec.cpp/build/tts-cli")
    ap.add_argument("--quant", default="q8_0")
    ap.add_argument("--zonos-repo", default="Zyphra/ZONOS3-Mini")
    ap.add_argument("--out", default="tts-wav")
    ap.add_argument("--models", default="models")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out, dest = Path(a.out), Path(a.models)
    out.mkdir(parents=True, exist_ok=True)
    dest.mkdir(exist_ok=True)

    # Extrait de référence : première phrase MLS de 6 à 12 s.
    ref = out / "reference_mls.wav"
    cand = [x for _, x in mls_french(12, dest) if 6 <= len(x) / 16000 <= 12]
    save(ref, cand[0], 16000)

    asr = Parakeet(dest)
    t0 = time.perf_counter()
    if a.engine == "chatterbox":
        eng = Chatterbox(ref)
    elif a.engine == "chatterbox-cpp":
        eng = ChatterboxCpp(ref, Path(a.tts_cli), dest, a.quant)
    else:
        eng = Zonos(ref, a.zonos_repo)
    print(f"{a.engine} chargé en {time.perf_counter() - t0:.0f} s", flush=True)

    results = []
    for label, text, opts in eng.jobs():
        try:
            t0 = time.perf_counter()
            w = eng(text, opts)
            dt = time.perf_counter() - t0
            dur = len(w) / eng.sr
            save(out / f"{a.engine}_{label}.wav", w / max(1e-6, np.abs(w).max()) * 0.95, eng.sr)
            hyp = asr(resample(w, eng.sr))
            r = {"fichier": f"{a.engine}_{label}.wav", "texte": text, "options": opts,
                 "duree_s": round(dur, 2), "calcul_s": round(dt, 1), "facteur_temps_reel": round(dt / dur, 2),
                 "wer": round(wer([text], [hyp]), 3), "transcription": normalize_fr(hyp)}
        except Exception:
            traceback.print_exc()
            r = {"fichier": None, "texte": text, "erreur": traceback.format_exc(limit=2)}
        results.append(r)
        print(f"{label:<26} {r.get('duree_s', '?')} s audio, temps réel x{r.get('facteur_temps_reel', '?')}, "
              f"WER {r.get('wer', '?')}\n    entendu : {r.get('transcription', '')}", flush=True)

    print("RESUME_TTS", json.dumps({r["fichier"]: [r.get("facteur_temps_reel"), r.get("wer")] for r in results},
                                   ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
