"""Protocole de jetons d'Omni-FR.

Trois choix fixés ici, parce qu'ils sont coûteux à changer après l'entraînement :

1. **Jetons spéciaux réservés dès maintenant** (``SPECIAL_TOKENS``), y compris des emplacements libres.
2. **Appels d'outils en syntaxe compacte**, pas en JSON. Le modèle émet un jeton par trame audio
   (12,5 Hz) ; un appel JSON de 40 jetons prendrait 3,2 s à ce rythme. Les appels sont donc émis en
   *rafale* sur des positions de contrôle (``PosType.CTRL``) qui n'avancent pas l'horloge audio, dans
   une syntaxe courte que l'orchestrateur traduit en JSON :

       <appel 1> meteo.prevision ville="Bordeaux" jour="demain" </appel>

3. **Historique en texte seul** : un tour clos ne garde que son texte, ses appels et ses résultats.
   Un tour de l'assistant interrompu est tronqué au dernier mot réellement joué.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Iterable, Sequence

SPECIAL_TOKENS: list[str] = [
    "<pad>",            # aucun texte sur cette trame
    "<bos>", "<eos>",
    "<tour_user>", "<tour_assistant>", "<fin_tour>",
    "<interrompu>",     # l'utilisateur a coupé l'assistant à cet endroit
    "<appel>", "</appel>",
    "<resultat>", "</resultat>",
    "<ok>", "<attente>", "<erreur>", "<refus>",
    "<confirmer>",      # action sensible : demander confirmation avant d'appeler
    "<repeter>",        # incompréhension : faire répéter plutôt qu'inventer
    "<silence>",
    "<outils>", "</outils>",
    "<entites>", "</entites>",
] + [f"<reserve_{i}>" for i in range(12)]

PAD = "<pad>"


class PosType(IntEnum):
    FRAME = 0   # une trame audio de 80 ms : audio entrant, texte, codes de la voix
    CTRL = 1    # position de rafale : jetons d'appel d'outil, l'horloge audio ne bouge pas


# ---------------------------------------------------------------------------
# Appels et résultats d'outils
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    call_id: int
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_compact(self) -> str:
        parts = [f"<appel {self.call_id}>", self.name]
        for k, v in self.args.items():
            parts.append(f"{k}={_fmt_value(v)}")
        parts.append("</appel>")
        return " ".join(parts)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.call_id, "name": self.name, "arguments": self.args}


@dataclass
class ToolResult:
    call_id: int
    status: str            # "ok" | "attente" | "erreur" | "refus"
    payload: Any = None

    def render(self) -> str:
        body = json.dumps(self.payload, ensure_ascii=False, separators=(",", ":"))
        return f"<resultat {self.call_id} <{self.status}>> {body} </resultat>"


_NAME_RE = r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*"
_CALL_RE = re.compile(rf"^<appel (\d+)>\s+({_NAME_RE})(.*?)\s*</appel>$", re.S)
_ARG_RE = re.compile(r'([a-z_][a-z0-9_]*)=("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|vrai|faux|nul)')


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "vrai" if v else "faux"
    if v is None:
        return "nul"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v), ensure_ascii=False)


def _parse_value(s: str) -> Any:
    if s == "vrai":
        return True
    if s == "faux":
        return False
    if s == "nul":
        return None
    if s.startswith('"'):
        return json.loads(s)
    return float(s) if "." in s else int(s)


class CallSyntaxError(ValueError):
    pass


def parse_call(text: str) -> ToolCall:
    """Analyse un appel compact. Lève ``CallSyntaxError`` si la syntaxe est invalide.

    En production, le décodage contraint rend cette erreur impossible ; ici elle sert à
    filtrer les données et à évaluer un modèle non contraint.
    """
    m = _CALL_RE.match(text.strip())
    if not m:
        raise CallSyntaxError(f"appel invalide : {text!r}")
    call_id, name, rest = int(m.group(1)), m.group(2), m.group(3)
    args: dict[str, Any] = {}
    pos = 0
    while pos < len(rest):
        if rest[pos].isspace():
            pos += 1
            continue
        am = _ARG_RE.match(rest, pos)
        if not am:
            raise CallSyntaxError(f"argument invalide à la position {pos} : {rest[pos:]!r}")
        if am.group(1) in args:
            raise CallSyntaxError(f"argument en double : {am.group(1)}")
        args[am.group(1)] = _parse_value(am.group(2))
        pos = am.end()
    return ToolCall(call_id, name, args)


# ---------------------------------------------------------------------------
# Historique conversationnel
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: str                                  # "user" | "assistant"
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)
    interrupted: bool = False


def render_history(turns: Sequence[Turn], tools_block: str = "", entities_block: str = "") -> str:
    """Rend l'historique en texte seul, tel que le backbone le voit au début d'un tour."""
    out: list[str] = ["<bos>"]
    if tools_block:
        out.append(f"<outils> {tools_block} </outils>")
    if entities_block:
        out.append(f"<entites> {entities_block} </entites>")
    for t in turns:
        out.append("<tour_user>" if t.role == "user" else "<tour_assistant>")
        out.extend(c.to_compact() for c in t.calls)
        out.extend(r.render() for r in t.results)
        if t.text:
            out.append(t.text)
        if t.interrupted:
            out.append("<interrompu>")
        out.append("<fin_tour>")
    return " ".join(out)


@dataclass
class WordTiming:
    word: str
    start: float   # secondes
    end: float


def truncate_to_played(words: Sequence[WordTiming], played_s: float) -> tuple[str, bool]:
    """Garde les mots entièrement joués avant la coupure.

    ``played_s`` vient du lecteur audio (compteur d'échantillons réellement envoyés au
    haut-parleur), jamais du modèle. Renvoie le texte gardé et ``True`` si la réponse a été
    coupée avant sa fin.
    """
    kept = [w.word for w in words if w.end <= played_s]
    interrupted = len(kept) < len(words)
    return " ".join(kept), interrupted


# ---------------------------------------------------------------------------
# Alignement du flux texte sur les trames audio
# ---------------------------------------------------------------------------

def align_text_to_frames(
    words: Sequence[WordTiming],
    n_frames: int,
    tokenize: Callable[[str], list[int]],
    pad_id: int,
    frame_hz: float = 12.5,
    lead_frames: int = 0,
) -> tuple[list[int], int]:
    """Place les jetons de chaque mot sur la grille de trames, un jeton au plus par trame.

    Chaque mot commence à la trame de son début acoustique, avancée de ``lead_frames`` (le
    texte de l'assistant précède sa voix). Si la trame est prise, les jetons glissent vers la
    droite. Renvoie la séquence et le retard maximal accumulé, en trames : un retard qui
    grandit signale un débit de parole trop rapide pour 12,5 jetons/s.
    """
    seq = [pad_id] * n_frames
    cursor = 0
    max_lag = 0
    for w in words:
        target = max(0, math.floor(w.start * frame_hz) - lead_frames)
        pos = max(cursor, target)
        max_lag = max(max_lag, pos - target)
        first = all(t == pad_id for t in seq[:pos])
        for tok in tokenize(w.word if first else " " + w.word):
            if pos >= n_frames:
                return seq, max_lag
            seq[pos] = tok
            pos += 1
        cursor = pos
    return seq, max_lag


def frames_for_seconds(seconds: float, frame_hz: float = 12.5) -> int:
    return math.ceil(seconds * frame_hz)


def iter_special(tokens: Iterable[str]) -> list[str]:
    return [t for t in tokens if t in SPECIAL_TOKENS]
