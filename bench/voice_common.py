"""Outils communs aux bancs vocaux : audio, bruit, jeu MLS français, transcription Parakeet (sherpa-onnx).

Aucune dépendance PyTorch ici : numpy, soundfile, huggingface_hub, sherpa-onnx.
"""

from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from bench.eval_audio import mix, pink, read_wav, resample, wav_bytes  # noqa: F401  (réexport)

PARAKEET_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
                "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2")

# Conditions retenues : propre, brouhaha (pièce où d'autres parlent) et bruit rose.
CONDITIONS = [("propre", None, "propre"), ("brouhaha", 10, "brouhaha 10 dB"), ("brouhaha", 0, "brouhaha 0 dB"),
              ("rose", 5, "rose 5 dB")]


def load_any(p: Path) -> np.ndarray:
    """Lit un fichier audio (wav/flac…) en mono 16 kHz float32."""
    try:
        import soundfile as sf
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = x.mean(1)
    except Exception:
        x, sr = read_wav(p)
    return resample(x.astype(np.float32), sr)


def mls_french(n: int, dest: Path) -> list[tuple[str, np.ndarray]]:
    """n phrases humaines réelles (Multilingual LibriSpeech, français, test), 16 kHz.

    MLS = lectures LibriVox (domaine public), transcriptions en minuscules sans ponctuation.
    On télécharge un seul fichier parquet, pas le jeu entier.
    """
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    repo = "facebook/multilingual_librispeech"
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
             if "french" in f and "test" in f and f.endswith(".parquet")]
    if not files:
        raise RuntimeError("aucun parquet MLS français trouvé")
    path = hf_hub_download(repo, sorted(files)[0], repo_type="dataset", local_dir=dest)
    table = pq.read_table(path)
    cols = table.column_names
    tcol = next(c for c in ("transcript", "text", "sentence") if c in cols)
    out, per_spk = [], {}
    # Varier les locuteurs : au plus 3 phrases par locuteur, de 4 à 16 s.
    for r in table.to_pylist():
        spk = r.get("speaker_id", r.get("speaker", ""))
        if per_spk.get(spk, 0) >= 3:
            continue
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        x = resample(x.mean(1), sr)
        if not 4 <= len(x) / 16000 <= 16:
            continue
        per_spk[spk] = per_spk.get(spk, 0) + 1
        out.append((r[tcol], x))
        if len(out) >= n:
            break
    return out


def noisy(clean: list[np.ndarray], i: int, kind: str | None, snr: float | None,
          rng: np.random.Generator) -> np.ndarray:
    x = clean[i]
    if snr is None:
        return x
    if kind == "rose":
        return mix(x, pink(len(x), rng), snr)
    others = [clean[(i + k) % len(clean)] for k in (1, 2, 3)]
    return mix(x, sum(np.resize(o, len(x)) for o in others), snr)


class Parakeet:
    """parakeet-tdt-0.6b-v3 en int8 via sherpa-onnx (ONNX Runtime, CPU). 25 langues dont le français."""

    name = "parakeet-v3-int8"

    def __init__(self, dest: Path, threads: int = 4):
        import sherpa_onnx
        d = dest / "parakeet"
        if not any(d.glob("**/tokens.txt")):
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "m.tar.bz2"
            urllib.request.urlretrieve(PARAKEET_URL, tmp)
            with tarfile.open(tmp) as t:
                t.extractall(d, filter="data")
            tmp.unlink()
        root = next(d.glob("**/tokens.txt")).parent
        f = lambda pat: str(next(root.glob(pat)))  # noqa: E731
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f("encoder*.onnx"), decoder=f("decoder*.onnx"), joiner=f("joiner*.onnx"),
            tokens=f("tokens.txt"), model_type="nemo_transducer", num_threads=threads)

    def __call__(self, x: np.ndarray) -> str:
        s = self.rec.create_stream()
        s.accept_waveform(16000, x)
        self.rec.decode_stream(s)
        return s.result.text.strip()
