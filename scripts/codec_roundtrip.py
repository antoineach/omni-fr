"""Expérience 1 du projet : le codec restitue-t-il le français et votre voix ?

Pour chaque fichier audio : encode, décode à 16 puis à 8 codebooks, transcrit l'original et
les deux reconstructions avec le même ASR, et mesure la similarité de locuteur.
Porte (PLAN.md) : écart de WER < 1 point à 16 codebooks, et décision 8/16 par écoute à l'aveugle.

    python scripts/codec_roundtrip.py --audio dossier_wav/ --out sorties/codec \
        --asr openai/whisper-large-v3 --device cuda

Entrées conseillées : 30 extraits Common Voice FR (validated), 30 extraits MLS FR, votre extrait
de référence découpé en phrases. Mono, n'importe quel taux d'échantillonnage.
Sorties : WAV reconstruits (pour l'écoute) + rapport JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnifr.metrics import wer  # noqa: E402


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def make_asr(model_id: str, device: str):
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=model_id, device=device)

    def run(wav: np.ndarray, sr: int) -> str:
        out = asr({"raw": wav, "sampling_rate": sr},
                  generate_kwargs={"language": "french", "task": "transcribe"})
        return out["text"]
    return run


def make_speaker(device: str):
    """Similarité cosinus ECAPA (speechbrain). Optionnel : renvoie None si absent."""
    try:
        import torch
        import torchaudio
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        print("speechbrain absent : similarité de locuteur ignorée (pip install speechbrain)")
        return None
    enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})

    def emb(wav, sr):
        t = torch.tensor(wav)[None]
        if sr != 16000:
            t = torchaudio.functional.resample(t, sr, 16000)
        return enc.encode_batch(t.to(device)).squeeze().cpu()

    def sim(a, sra, b, srb) -> float:
        import torch.nn.functional as F
        return float(F.cosine_similarity(emb(a, sra), emb(b, srb), dim=0))
    return sim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="dossier de .wav/.flac")
    ap.add_argument("--out", default="sorties/codec")
    ap.add_argument("--asr", default="openai/whisper-large-v3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--codebooks", default="16,8", help="niveaux à tester, séparés par des virgules")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from omnifr.codec import QwenCodec

    files = sorted(p for p in Path(a.audio).iterdir() if p.suffix.lower() in (".wav", ".flac"))
    if a.limit:
        files = files[: a.limit]
    if not files:
        sys.exit(f"aucun fichier audio dans {a.audio}")
    levels = [int(x) for x in a.codebooks.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    codec = QwenCodec(device=a.device)
    asr = make_asr(a.asr, a.device)
    spk = make_speaker(a.device)

    ref_txt: list[str] = []
    hyp_txt: dict[int, list[str]] = {k: [] for k in levels}
    sims: dict[int, list[float]] = {k: [] for k in levels}
    rows = []
    for f in files:
        wav, sr = load_mono(f)
        codes = codec.encode(wav, sr)
        t_ref = asr(wav, sr)
        ref_txt.append(t_ref)
        row = {"fichier": f.name, "trames": int(codes.shape[0]), "original": t_ref}
        for k in levels:
            rec, sro = codec.decode(codes, n_codebooks=k)
            sf.write(out / f"{f.stem}_k{k}.wav", rec, sro)
            t = asr(rec, sro)
            hyp_txt[k].append(t)
            row[f"k{k}"] = t
            if spk:
                s = spk(wav, sr, rec, sro)
                sims[k].append(s)
                row[f"sim_k{k}"] = round(s, 4)
        rows.append(row)
        print(f"{f.name}: {codes.shape[0]} trames")

    report = {
        "asr": a.asr,
        "n_fichiers": len(files),
        # WER des reconstructions *par rapport à la transcription de l'original* par le même ASR :
        # mesure ce que le codec détruit, indépendamment des erreurs propres de l'ASR.
        "wer_vs_original": {f"k{k}": round(wer(ref_txt, hyp_txt[k]), 4) for k in levels},
        "similarite_locuteur": {f"k{k}": round(float(np.mean(v)), 4) for k, v in sims.items() if v},
        "details": rows,
    }
    (out / "rapport.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
