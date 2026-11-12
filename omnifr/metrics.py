"""Métriques sans dépendance lourde."""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s'-]", re.UNICODE)


def normalize_fr(text: str) -> str:
    """Normalisation minimale pour le WER : minuscules, apostrophes unifiées, sans ponctuation.

    Les nombres ne sont pas convertis en lettres : comparer des transcriptions produites par le
    même ASR, ou normaliser les nombres en amont.
    """
    t = unicodedata.normalize("NFC", text).lower().replace("’", "'").replace("‘", "'")
    t = _PUNCT.sub(" ", t)
    t = re.sub(r"(\w)'(\w)", r"\1' \2", t)
    return " ".join(t.split())


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer(refs: list[str], hyps: list[str]) -> float:
    """WER cumulé sur un corpus (erreurs totales / mots de référence totaux)."""
    err = n = 0
    for r, h in zip(refs, hyps, strict=True):
        rw, hw = normalize_fr(r).split(), normalize_fr(h).split()
        err += edit_distance(rw, hw)
        n += len(rw)
    return err / max(1, n)
