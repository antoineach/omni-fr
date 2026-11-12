"""Banc de vitesse temps réel de l'étudiant, à exécuter sur la machine cible ou un proxy.

Mesure, poids aléatoires (la vitesse ne dépend pas des valeurs) :
- un pas de backbone FRAME avec un contexte déjà rempli (historique + tour en cours) ;
- un pas d'encodeur audio ;
- la génération des K codes d'une trame par le Depth Transformer.

Une trame dure 80 ms. Le script rapporte la part de ce budget consommée par trame :
au-delà de ~70 %, il ne reste pas de marge pour le codec, l'annulation d'écho et l'OS.

Exemples :
    python -m bench.bench_step --preset omni500 --codebooks 8 --threads 4
    python -m bench.bench_step --preset all --threads 4 --int8 --json resultats.json

``--int8`` applique une quantification dynamique int8 des couches linéaires (PyTorch), proche
de ce qu'un déploiement donnerait ; la quantification 4 bits demande un autre moteur.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch

from omnifr.config import PRESETS, preset
from omnifr.model import OmniStudent, count_params

FRAME_MS = 80.0


def _quantize(model: torch.nn.Module) -> torch.nn.Module:
    engines = torch.backends.quantized.supported_engines
    if "qnnpack" in engines and platform.machine().lower() in ("arm64", "aarch64"):
        torch.backends.quantized.engine = "qnnpack"
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def _timed(fn, n: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


@torch.no_grad()
def bench_preset(name: str, codebooks: int | None, context: int, n: int, int8: bool) -> dict:
    cfg = preset(name)
    if codebooks:
        cfg = cfg.with_codebooks(codebooks)
    model = OmniStudent(cfg).eval()
    # le cache ne doit pas dépasser la table RoPE du préréglage (512 pour « tiny »)
    context = min(context, cfg.backbone.max_len - n - 10)
    params = count_params(model)
    if int8:
        model = _quantize(model)

    state = model.new_state()
    d = model.backbone.dim
    # Pré-remplit le cache du backbone avec ``context`` positions (historique texte + trames).
    model.backbone(torch.randn(1, context, d), state["bb"])
    # Pré-remplit l'encodeur avec ~2 s d'audio.
    model.encoder(torch.randn(1, 200, cfg.encoder.n_mels), state["enc"])

    win = cfg.encoder.n_subsample * 2 ** cfg.encoder.n_subsample   # = AudioEncoder.step_window
    mel = torch.randn(1, win, cfg.encoder.n_mels)
    tok = torch.zeros(1, dtype=torch.long)
    audio = torch.zeros(1, cfg.n_codebooks, dtype=torch.long)

    # Chaque mesure ajoute une position au cache : on borne n pour rester proche de ``context``.
    enc_ms = _timed(lambda: model.encoder.step(mel, state["enc"]), n, 3)
    h_box = {}

    def frame():
        _, h = model.frame_step(state, mel, tok, audio)
        h_box["h"] = h

    frame_ms = _timed(frame, n, 3)
    bb_ms = [f - e for f, e in zip(frame_ms, enc_ms)]
    depth_ms = _timed(lambda: model.depth.generate(h_box["h"], tok), n, 3)

    med = statistics.median
    total = med(frame_ms) + med(depth_ms)
    return {
        "preset": cfg.name,
        "codebooks": cfg.n_codebooks,
        "int8": int8,
        "threads": torch.get_num_threads(),
        "context": context,
        "params_M": {k: round(v / 1e6, 1) for k, v in params.items()},
        "ms_par_trame": {
            "encodeur": round(med(enc_ms), 2),
            "backbone (approx.)": round(med(bb_ms), 2),
            "depth": round(med(depth_ms), 2),
            "total en parole": round(total, 2),
        },
        "part_du_budget_80ms": round(total / FRAME_MS, 3),
        "p95_total_ms": round(sorted(f + d_ for f, d_ in zip(frame_ms, depth_ms))[int(0.95 * n) - 1], 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="omni500", help=f"un de {PRESETS} ou 'all'")
    ap.add_argument("--codebooks", type=int, default=0, help="0 = valeur du préréglage")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--context", type=int, default=1500, help="positions déjà dans le cache")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    # « tiny » mesure le plancher de surcoût Python/PyTorch par trame (calcul quasi nul) :
    # ce qu'un moteur compilé (C++, MLX, ONNX) supprimerait.
    names = list(PRESETS) if a.preset == "all" else [a.preset]
    host = {"machine": platform.machine(), "processor": platform.processor(),
            "system": platform.platform(), "torch": torch.__version__}
    results = []
    for name in names:
        r = bench_preset(name, a.codebooks or None, a.context, a.n, a.int8)
        results.append(r)
        print(f"{r['preset']:<26} K={r['codebooks']:<2} int8={r['int8']!s:<5} "
              f"total {r['ms_par_trame']['total en parole']:7.2f} ms/trame "
              f"= {100 * r['part_du_budget_80ms']:5.1f} % du temps réel  "
              f"(p95 {r['p95_total_ms']} ms)  "
              + " ".join(f"{k}={v}" for k, v in r["ms_par_trame"].items() if k != "total en parole"))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"hote": host, "resultats": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
