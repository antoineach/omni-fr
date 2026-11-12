"""Coût du décodeur du codec en streaming : le poste qui peut à lui seul exclure le Pi 5.

Mesure le décodage d'un bloc de quelques trames, précédé de ``--context`` trames de contexte
recalculées à chaque bloc (la méthode de ``chunked_decode`` de Qwen). C'est pessimiste pour un
décodeur à état, mais identique pour les deux codecs : la comparaison reste juste.

    python -m bench.bench_codec --codec qwen --threads 4 --chunk 4
    python -m bench.bench_codec --codec mimi --threads 4 --chunk 4
    python -m bench.bench_codec --codec qwen --random     # sans téléchargement

Premier résultat (runner GitHub ubuntu-24.04-arm, 3 threads, 12 nov. 2026) : décodeur Qwen réel
= 114,3 M de paramètres, 3,7 fois le temps réel en blocs de 320 ms, 1,5 fois en blocs de 960 ms.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch


def load_decoder(codec: str, random_init: bool, n_codebooks: int):
    """Renvoie (fonction de décodage sur codes (1, K, T), nb de paramètres, K, source)."""
    if codec == "mimi":
        from transformers import MimiConfig, MimiModel
        if random_init:
            m, src = MimiModel(MimiConfig()).eval(), "Mimi, config par défaut, poids aléatoires"
        else:
            m, src = MimiModel.from_pretrained("kyutai/mimi").eval(), "checkpoint kyutai/mimi"
        k = n_codebooks or 8
        n = sum(p.numel() for name, p in m.named_parameters()
                if name.startswith(("decoder", "upsample", "decoder_transformer", "quantizer")))
        return (lambda c: m.decode(c)), n, k, src
    if random_init:
        from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
        cfg = Qwen3TTSTokenizerV2Config()
        if not hasattr(cfg.decoder_config, "codebook_dim"):
            cfg.decoder_config.codebook_dim = 512   # absent de la config par défaut
        dec, src = Qwen3TTSTokenizerV2Model(cfg).eval().decoder, "Qwen, config par défaut, poids aléatoires"
    else:
        from omnifr.codec import QwenCodec
        dec, src = QwenCodec().model.decoder, "checkpoint Qwen/Qwen3-TTS-Tokenizer-12Hz"
    return dec, sum(p.numel() for p in dec.parameters()), dec.config.num_quantizers, src


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", choices=["qwen", "mimi"], default="qwen")
    ap.add_argument("--codebooks", type=int, default=0, help="Mimi seulement (défaut 8)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4, help="trames décodées par bloc (4 = 320 ms)")
    ap.add_argument("--context", type=int, default=25, help="trames de contexte gauche, comme Qwen")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    dec, n_params, K, source = load_decoder(a.codec, a.random, a.codebooks)
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
    res = {"codec": a.codec, "source": source, "machine": platform.machine(), "threads": a.threads,
           "codebooks": K, "decodeur_M": round(n_params / 1e6, 1), "bloc_trames": a.chunk,
           "ms_par_bloc": round(ms, 1), "part_du_temps_reel": round(ms / budget, 3)}
    print(json.dumps(res, ensure_ascii=False))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
