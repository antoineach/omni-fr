"""Métriques sans dépendance lourde."""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s'-]", re.UNICODE)


_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers_to_words(t: str) -> str:
    """« 30 % » -> « trente pour cent », « 21,5 » -> « vingt et un virgule cinq ».

    Sans num2words installé, les chiffres restent tels quels.
    """
    t = t.replace("%", " pour cent ").replace("°c", " degrés ").replace("°", " degrés ")
    try:
        from num2words import num2words
    except ImportError:
        return t

    def conv(m: re.Match) -> str:
        x = m.group(0).replace(",", ".")
        try:
            v = float(x) if "." in x else int(x)
            return " " + num2words(v, lang="fr").replace("virgule zéro", "").strip() + " "
        except Exception:
            return m.group(0)
    t = re.sub(r"(\d+)\s*h\s*(\d+)", r"\1 heures \2", t)
    return _NUM.sub(conv, t)


def normalize_fr(text: str, numbers: bool = True) -> str:
    """Normalisation pour le WER : minuscules, apostrophes unifiées, traits d'union et
    ponctuation retirés, nombres écrits en lettres (si num2words est installé)."""
    t = unicodedata.normalize("NFC", text).lower().replace("’", "'").replace("‘", "'")
    if numbers:
        t = _numbers_to_words(t)
    t = t.replace("-", " ")
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
