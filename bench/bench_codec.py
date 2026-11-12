"""Coût du décodeur du codec en streaming : le poste qui peut à lui seul exclure le Pi 5.

Avec la configuration par défaut de la classe (poids aléatoires), le décodeur Qwen3-TTS 12 Hz
compte ~195 M de paramètres et décode 10 s d'audio en ~5 s sur 2 cœurs Xeon. Ce script mesure
le vrai checkpoint, en blocs de quelques trames comme en conversation.

    python -m bench.bench_codec --threads 4 --chunk 4
    python -m bench.bench_codec --random     # sans téléchargement, config par défaut

Sortie : paramètres du décodeur, ms par bloc, part du temps réel.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch


def load_decoder(random_init: bool):
    if random_init:
        from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
        cfg = Qwen3TTSTokenizerV2Config()
        if not hasattr(cfg.decoder_config, "codebook_dim"):
            cfg.decoder_config.codebook_dim = 512   # absent de la config par défaut
        return Qwen3TTSTokenizerV2Model(cfg).eval().decoder, "config par défaut, poids aléatoires"
    from omnifr.codec import QwenCodec
    c = QwenCodec()
    return c.model.decoder, "checkpoint Qwen/Qwen3-TTS-Tokenizer-12Hz"


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4, help="trames décodées par bloc (4 = 320 ms)")
    ap.add_argument("--context", type=int, default=25, help="trames de contexte gauche, comme Qwen")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    dec, source = load_decoder(a.random)
    n_params = sum(p.numel() for p in dec.parameters())
    K = dec.config.num_quantizers
    codes = torch.randint(0, 2048, (1, K, a.context + a.chunk))
    for _ in range(3):
        dec(codes)
    times = []
    for _ in range(a.n):
        t0 = time.perf_counter()
        dec(codes)
        times.append((time.perf_counter() - t0) * 1000)
    ms = statistics.median(times)
    budget = a.chunk * 80.0
    res = {"source": source, "machine": platform.machine(), "threads": a.threads,
           "decodeur_M": round(n_params / 1e6, 1), "bloc_trames": a.chunk,
           "ms_par_bloc": round(ms, 1), "part_du_temps_reel": round(ms / budget, 3)}
    print(json.dumps(res, ensure_ascii=False))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
