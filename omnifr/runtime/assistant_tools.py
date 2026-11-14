"""Outils du quotidien : heure et date, minuteurs, calculs exacts.

Ils s'ajoutent au simulateur (qui valide les arguments) et s'exécutent localement, sans réseau.
La recherche web, elle, est dans ``backends.py`` (SearXNG, ou Wikipédia sans configuration).
"""

from __future__ import annotations

import ast
import math
import operator
import threading
import time
from datetime import datetime
from typing import Callable

from ..sim.home import HomeSimulator, ToolError, ToolSpec

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
        "novembre", "décembre"]


def maintenant(now: datetime | None = None) -> dict:
    n = now or datetime.now().astimezone()
    return {"heure": f"{n.hour} h {n.minute:02d}", "date": f"{JOURS[n.weekday()]} {n.day} {MOIS[n.month - 1]} {n.year}",
            "iso": n.isoformat(timespec="minutes"), "fuseau": str(n.tzinfo)}


def _duree(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    parts = ([f"{h} h"] if h else []) + ([f"{m} min"] if m else []) + ([f"{sec} s"] if sec or not (h or m) else [])
    return " ".join(parts)


class Timers:
    """Minuteurs en fil d'exécution ; ``on_ring(nom)`` est appelé à l'échéance."""

    def __init__(self, on_ring: Callable[[str], None] | None = None):
        self.on_ring = on_ring or (lambda nom: print(f"\n⏰ Minuteur « {nom} » terminé !", flush=True))
        self.active: dict[str, tuple[float, threading.Timer]] = {}
        self.lock = threading.Lock()

    def start(self, a: dict) -> dict:
        nom = (a.get("nom") or "minuteur").strip().lower()
        d = float(a["duree_s"])
        with self.lock:
            if nom in self.active:
                self.active.pop(nom)[1].cancel()
            t = threading.Timer(d, self._ring, args=(nom,))
            t.daemon = True
            self.active[nom] = (time.time() + d, t)
            t.start()
        return {"nom": nom, "duree": _duree(d), "fin_vers": maintenant(datetime.now().astimezone()
                                                                         .fromtimestamp(time.time() + d)
                                                                         .astimezone())["heure"]}

    def _ring(self, nom: str) -> None:
        with self.lock:
            self.active.pop(nom, None)
        self.on_ring(nom)

    def list(self, a: dict) -> dict:
        now = time.time()
        with self.lock:
            return {"minuteurs": [{"nom": n, "reste": _duree(end - now)} for n, (end, _) in self.active.items()]}

    def cancel(self, a: dict) -> dict:
        nom = (a.get("nom") or "").strip().lower()
        with self.lock:
            if nom not in self.active:
                if len(self.active) == 1 and not nom:
                    nom = next(iter(self.active))
                else:
                    raise ToolError("introuvable", f"aucun minuteur nommé « {nom} »")
            self.active.pop(nom)[1].cancel()
        return {"annule": nom}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "racine": math.sqrt, "abs": abs, "round": round, "arrondi": round, "log": math.log,
          "ln": math.log, "log10": math.log10, "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "min": min, "max": max, "floor": math.floor, "ceil": math.ceil}
_CONSTS = {"pi": math.pi, "e": math.e}


def calcul(expression: str) -> float:
    """Évalue une expression arithmétique sans ``eval`` (arbre syntaxique filtré)."""
    import re
    expr = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"(\d),(\d)", r"\1.\2", expr)                  # 2,5 -> 2.5
    expr = re.sub(r"(?<=[\d)])\s*[xX]\s*(?=[\d(])", "*", expr)     # 3 x 4 -> 3*4
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*(de|of|\*)\s*", r"\1/100*", expr)   # 15 % de 80
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 100:
                raise ToolError("trop_grand", "exposant trop grand")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _CONSTS:
            return _CONSTS[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(x) for x in n.args])
        raise ToolError("expression_invalide", "expression non reconnue")
    try:
        v = ev(ast.parse(expr, mode="eval"))
    except ToolError:
        raise
    except ZeroDivisionError:
        raise ToolError("division_par_zero", "division par zéro") from None
    except Exception:
        raise ToolError("expression_invalide", f"expression non reconnue : {expression}") from None
    return round(v, 10) if isinstance(v, float) else v


def register_extra(sim: HomeSimulator, timers: Timers) -> None:
    sim.register(ToolSpec("temps.maintenant", "Donne l'heure et la date actuelles", {}, kind="lecture"),
                 lambda a: maintenant())
    sim.register(ToolSpec("minuteur.lancer", "Lance un minuteur (durée en secondes, nom facultatif, ex. pâtes)",
                          {"duree_s": ("int", True), "nom": ("str", False)}, ranges={"duree_s": (1, 86400)}),
                 timers.start)
    sim.register(ToolSpec("minuteur.liste", "Liste les minuteurs en cours et le temps restant", {}, kind="lecture"),
                 timers.list)
    sim.register(ToolSpec("minuteur.annuler", "Annule un minuteur par son nom",
                          {"nom": ("str", False)}), timers.cancel)
    sim.register(ToolSpec("calcul.evaluer", "Calcule exactement une expression arithmétique (+ - * / ** sqrt % …)",
                          {"expression": ("str", True)}, kind="lecture"),
                 lambda a: {"expression": a["expression"], "resultat": calcul(a["expression"])})


RULES_EXTRA = """- Heure ou date : appelle temps.maintenant, ne devine jamais.
- Minuteur : convertis la durée en secondes (« 10 minutes » = 600) et appelle minuteur.lancer avec un nom court (pâtes, four…).
- Calcul de plus d'une opération simple, pourcentage, conversion : appelle calcul.evaluer et dis le résultat arrondi raisonnablement.
- Fait sur une personne, une date, un lieu, un événement (date de naissance, capitale, record…) que tu ne connais pas avec certitude : appelle web.recherche."""
