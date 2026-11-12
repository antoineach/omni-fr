"""Jeu de scénarios domotiques français et notation des appels d'outils.

Format JSONL, une ligne par scénario :

    {"id": "par-01", "categorie": "parallele",
     "dialogue": [{"role": "user", "texte": "..."}, ...],
     "attendu": {"appels": [{"name": "...", "args": {...}}],
                 "comportement": "appeler" | "confirmer" | "repeter" | "repondre"}}

Le dernier tour utilisateur est celui à traiter. Les métriques suivent la section 11 de l'étude :
F1 de sélection d'outil, exactitude stricte des arguments, réussite complète (Pass@1).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

BEHAVIOURS = ("appeler", "confirmer", "repeter", "repondre")


@dataclass
class Scenario:
    id: str
    categorie: str
    dialogue: list[dict[str, str]]
    appels: list[dict[str, Any]]
    comportement: str
    reponse_contient: list[list[str]] = field(default_factory=list)   # ET de OU, insensible à la casse

    @property
    def parallel(self) -> bool:
        return len(self.appels) > 1


@dataclass
class Prediction:
    appels: list[dict[str, Any]] = field(default_factory=list)
    comportement: str = "repondre"
    texte: str = ""


def load(path: str | Path) -> list[Scenario]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            d = json.loads(line)
            beh = d["attendu"]["comportement"]
            if beh not in BEHAVIOURS:
                raise ValueError(f"{d['id']}: comportement inconnu {beh}")
            out.append(Scenario(d["id"], d["categorie"], d["dialogue"], d["attendu"].get("appels", []), beh,
                                d["attendu"].get("reponse_contient", [])))
    return out


def _norm(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)                      # 21 et 21.0 sont le même argument
    if isinstance(v, str):
        return v.strip().casefold()
    return v


def _canon(call: dict[str, Any], libre: bool = False) -> str:
    """Forme canonique d'un appel. ``libre`` : seul le nom compte (requête web en texte libre)."""
    args = {} if libre else {k: _norm(v) for k, v in call.get("args", {}).items()}
    return json.dumps({"name": call["name"], "args": args}, sort_keys=True, ensure_ascii=False)


def _contains(text: str, groups: list[list[str]]) -> bool:
    t = text.casefold()
    return all(any(alt.casefold() in t for alt in g) for g in groups)


def score_one(s: Scenario, p: Prediction) -> dict[str, Any]:
    exp_names, got_names = Counter(c["name"] for c in s.appels), Counter(c["name"] for c in p.appels)
    tp = sum((exp_names & got_names).values())
    prec = tp / max(1, sum(got_names.values())) if got_names else float(not exp_names)
    rec = tp / max(1, sum(exp_names.values())) if exp_names else float(not got_names)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    libres = {c["name"] for c in s.appels if c.get("args_libres")}
    args_exact = (Counter(_canon(c, c["name"] in libres) for c in s.appels)
                  == Counter(_canon(c, c["name"] in libres) for c in p.appels))
    contenu_ok = _contains(p.texte, s.reponse_contient) if s.reponse_contient else True
    beh_ok = p.comportement == s.comportement
    passed = args_exact and beh_ok and contenu_ok
    return {"id": s.id, "categorie": s.categorie, "f1_outil": f1, "args_exacts": args_exact,
            "comportement_ok": beh_ok, "contenu_ok": contenu_ok, "reussite": passed,
            "parallele": s.parallel}


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {}

    def mean(key, subset=None):
        r = [x for x in rows if subset is None or subset(x)]
        return round(sum(float(x[key]) for x in r) / len(r), 4) if r else None

    cats = sorted({r["categorie"] for r in rows})
    return {
        "n": len(rows),
        "f1_outil": mean("f1_outil"),
        "args_exacts": mean("args_exacts"),
        "reussite": mean("reussite"),
        "reussite_parallele": mean("reussite", lambda x: x["parallele"]),
        "par_categorie": {c: mean("reussite", lambda x, c=c: x["categorie"] == c) for c in cats},
    }
