"""Orchestrateur : le seul composant qui touche à la maison.

Le LLM propose des appels ; l'orchestrateur les vérifie et les exécute. Règles imposées par le
code, quel que soit le modèle (les bancs ont montré qu'aucun modèle ne les tient seul) :

1. Une action sensible (serrure, alarme, porte du garage) n'est jamais exécutée sans un « oui »
   explicite de l'utilisateur au tour suivant. L'orchestrateur garde l'appel en attente et
   l'exécute lui-même après le oui ; le modèle ne peut pas contourner la règle.
2. Une action sensible proposée *après* la lecture d'un résultat d'outil (page web, état…) est
   ignorée : c'est le motif d'une injection d'instruction.
3. Les appels écrits dans le texte (format natif Gemma 4, style Python) sont récupérés.
4. La réponse peut commencer par une étiquette d'émotion « [joie] » qui pilote la voix.

Le LLM est une simple fonction ``messages, tools -> message`` (format OpenAI), ce qui permet de
tester l'orchestrateur sans modèle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..protocol import ToolCall

SENSITIVE_TOOLS = {"maison.serrure", "maison.alarme"}
EMOTIONS = ("calme", "joie", "inquietude", "ironie", "neutre")
_EMO = re.compile(r"^\s*\[(calme|joie|inqui[ée]tude|ironie|neutre)\]\s*", re.I)
_YES = re.compile(r"^\s*(oui|ouais|vas[- ]y|confirm|d'accord|ok\b|okay|bien sûr|c'est bon|fais[- ]le|allez)", re.I)
_NO = re.compile(r"^\s*(non|annule|laisse|stop|pas maintenant|surtout pas|n'en fais rien)", re.I)

# Appels écrits en texte (format natif Gemma 4 « nom{k:<|"|>v<|"|>} » ou style Python « nom(k="v") »)
_TXT_CALL = re.compile(r"((?:assistant|maison|meteo|web)(?:\.|__)[a-z_]+)\s*([({])(.*?)([)}])", re.S)
_KV_PY = re.compile(r'([a-zA-Z_]\w*)\s*=\s*("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|True|False|true|false)')
_KV_GEMMA = re.compile(r'([a-zA-Z_]\w*)\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|true|false)')
_STRIP_CALLS = re.compile(r"(?:<\|?tool_call\|?>|<tool_call\|>)", re.S)

RULES_EMOTION = ("- Commence chaque réponse par une étiquette d'émotion entre crochets, choisie parmi "
                 "[calme] [joie] [inquietude] [ironie] [neutre], puis la phrase à dire.")


def api_name(n: str) -> str:
    return n.replace(".", "__")


def internal_name(n: str) -> str:
    return n.replace("__", ".")


def parse_text_calls(text: str) -> list[dict]:
    calls = []
    for m in _TXT_CALL.finditer(text.replace('<|"|>', '"')):
        name, body = internal_name(m.group(1)), m.group(3)
        args: dict[str, Any] = {}
        for k, v in (_KV_PY if m.group(2) == "(" else _KV_GEMMA).findall(body):
            if v.startswith('"'):
                args[k] = json.loads(v)
            elif v in ("True", "true"):
                args[k] = True
            elif v in ("False", "false"):
                args[k] = False
            else:
                args[k] = float(v) if "." in v else int(v)
        calls.append({"name": name, "args": args})
    return calls


def split_emotion(text: str) -> tuple[str, str]:
    m = _EMO.match(text or "")
    if not m:
        return "neutre", (text or "").strip()
    emo = m.group(1).lower().replace("é", "e")
    return emo, text[m.end():].strip()


class Backend(Protocol):
    """Ce qu'il faut pour exécuter des outils : le simulateur, ou Home Assistant réel."""

    tools: dict
    entities: dict

    def execute(self, call: ToolCall, confirmed: bool = False): ...


@dataclass
class Reply:
    text: str
    emotion: str = "neutre"
    actions: list[dict] = field(default_factory=list)       # appels exécutés et leurs résultats
    attend_confirmation: bool = False
    bloques: list[dict] = field(default_factory=list)        # appels refusés par les règles


LLM = Callable[[list[dict], list[dict]], dict]


class Orchestrator:
    def __init__(self, llm: LLM, backend: Backend, system: str, tools: list[dict],
                 max_rounds: int = 4, on_wait: Callable[[str], None] | None = None,
                 slow_tools: tuple[str, ...] = ("meteo.prevision", "web.recherche"),
                 max_history: int = 30):
        self.llm, self.backend, self.tools = llm, backend, tools
        self.system = system
        self.history: list[dict] = []
        self.max_rounds, self.on_wait, self.slow_tools = max_rounds, on_wait, slow_tools
        self.max_history = max_history
        self.trace: list[dict] = []            # chaque appel au LLM : contexte, sortie, tour (pour la distillation)
        self.pending: list[dict] = []          # appels sensibles en attente de « oui »
        self.asked_confirmation = False         # le modèle a demandé une confirmation sans appel précis
        self._n = 0

    # -- règles ---------------------------------------------------------------

    def is_sensitive(self, call: dict) -> bool:
        from ..sim.home import SENSITIVE_ENTITIES
        return call["name"] in SENSITIVE_TOOLS or call.get("args", {}).get("entite") in SENSITIVE_ENTITIES

    def _describe(self, calls: list[dict]) -> str:
        parts = []
        for c in calls:
            a = c.get("args", {})
            if c["name"] == "maison.serrure":
                parts.append("verrouille la porte d'entrée" if a.get("verrouiller") else "déverrouille la porte d'entrée")
            elif c["name"] == "maison.alarme":
                parts.append("arme l'alarme" if a.get("armer") else "désarme l'alarme")
            elif a.get("entite") == "cover.garage":
                pos = a.get("position", 100)
                parts.append("ferme la porte du garage" if pos == 0 else f"ouvre la porte du garage à {pos} %")
            else:
                parts.append(f"exécute {c['name']}")
        return " et que je ".join(parts)

    # -- exécution ------------------------------------------------------------

    def _exec(self, calls: list[dict], confirmed: bool) -> list[dict]:
        out = []
        if self.on_wait and any(c["name"] in self.slow_tools for c in calls):
            self.on_wait("Je regarde.")
        msg = {"role": "assistant", "content": "", "tool_calls": []}
        results = []
        for c in calls:
            self._n += 1
            msg["tool_calls"].append({"id": f"call_{self._n}", "type": "function", "function": {
                "name": api_name(c["name"]), "arguments": json.dumps(c.get("args", {}), ensure_ascii=False)}})
            res, _ = self.backend.execute(ToolCall(self._n, c["name"], c.get("args", {})), confirmed=confirmed)
            results.append({"role": "tool", "tool_call_id": f"call_{self._n}",
                            "content": json.dumps({"statut": res.status, "resultat": res.payload}, ensure_ascii=False)})
            out.append({"appel": c, "statut": res.status, "resultat": res.payload})
        self.history += [msg] + results
        return out

    def _calls_of(self, msg: dict) -> list[dict]:
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": internal_name(tc["function"]["name"]), "args": args})
        if not calls:
            calls = parse_text_calls(msg.get("content") or "")
        return calls

    def _finish(self, text: str, actions: list[dict], blocked: list[dict], ask: bool = False) -> Reply:
        emo, clean = split_emotion(text)
        clean = _STRIP_CALLS.sub("", clean).strip()
        # L'étiquette reste dans l'historique : le modèle voit le format qu'on attend de lui.
        self.history.append({"role": "assistant", "content": f"[{emo}] {clean}"})
        self.history = self.history[-self.max_history:]
        return Reply(clean, emo, actions, ask, blocked)

    # -- tour de parole ---------------------------------------------------------

    def turn(self, user_text: str) -> Reply:
        self.history.append({"role": "user", "content": user_text})
        actions: list[dict] = []
        allow_sensitive = False

        if self.pending or self.asked_confirmation:
            pending, self.pending = self.pending, []
            asked, self.asked_confirmation = self.asked_confirmation, False
            if _YES.search(user_text):
                if pending:
                    actions += self._exec(pending, confirmed=True)
                    ok = all(a["statut"] == "ok" for a in actions)
                    return self._finish("[calme] C'est fait." if ok else "[inquietude] L'action a échoué.", actions, [])
                allow_sensitive = asked     # le modèle avait demandé sans appel précis : il refait l'appel
            elif _NO.search(user_text):
                return self._finish("[calme] D'accord, je ne fais rien.", actions, [])

        blocked: list[dict] = []
        for rnd in range(self.max_rounds):
            ctx = [{"role": "system", "content": self.system}] + self.history
            msg = self.llm(ctx, self.tools)
            calls = self._calls_of(msg)
            self.trace.append({"contexte": [dict(m) for m in ctx], "sortie": msg, "appels": calls, "tour": rnd,
                               "confirmation_accordee": allow_sensitive})
            if not calls:
                return self._finish(msg.get("content") or "", actions, blocked)

            asks = [c for c in calls if c["name"] == "assistant.confirmer"]
            repeats = [c for c in calls if c["name"] == "assistant.faire_repeter"]
            real = [c for c in calls if not c["name"].startswith("assistant.")]
            sensitive = [c for c in real if self.is_sensitive(c)]
            safe = [c for c in real if not self.is_sensitive(c)]

            if sensitive:
                if rnd > 0:
                    blocked += sensitive        # après lecture d'un résultat : possible injection
                    sensitive = []
                    if not safe:
                        return self._finish("[calme] Le résultat contenait une demande d'action sensible ; "
                                            "je ne l'exécute pas sans que vous me le demandiez.", actions, blocked)
                elif allow_sensitive:
                    actions += self._exec(sensitive, confirmed=True)
                    sensitive = []
            if safe:
                actions += self._exec(safe, confirmed=False)
            if sensitive:
                self.pending = sensitive
                q = f"Vous confirmez que je {self._describe(sensitive)} ?"
                return self._finish(f"[calme] {q}", actions, blocked, ask=True)
            if asks and not safe:
                self.asked_confirmation = True
                q = asks[0]["args"].get("question") or "Vous confirmez ?"
                return self._finish(f"[calme] {q}", actions, blocked, ask=True)
            if repeats and not safe:
                q = repeats[0]["args"].get("question") or "Pouvez-vous répéter ?"
                return self._finish(f"[neutre] {q}", actions, blocked)
            if not safe and not actions:
                return self._finish(msg.get("content") or "", actions, blocked)
        return self._finish("[inquietude] Je n'ai pas réussi à terminer cette demande.", actions, blocked)
