"""Simulateur de maison : source unique de vérité pour tous les résultats d'outils.

Règle de l'usine à données : **aucun résultat d'outil n'est écrit par un LLM**. L'enseignant
propose des appels, ce simulateur les valide et les exécute ; ses réponses (y compris erreurs,
délais et refus) sont celles que l'étudiant apprend à lire.

Les noms d'outils imitent les intentions Home Assistant (exposées par son serveur MCP) et deux
outils externes. Les schémas sont volontairement petits ; les remplacer par les vrais schémas
exportés de votre instance Home Assistant dès la semaine 2.
"""

from __future__ import annotations

import copy
import random
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable

from ..protocol import ToolCall, ToolResult

# ---------------------------------------------------------------------------
# Schémas
# ---------------------------------------------------------------------------

ARG_TYPES = {"str": str, "int": int, "num": (int, float), "bool": bool}

# Entités dont toute commande exige une confirmation, quel que soit l'outil (portes, serrures).
SENSITIVE_ENTITIES = {"cover.garage", "lock.porte_entree"}


@dataclass
class ToolSpec:
    name: str
    description: str
    args: dict[str, tuple[str, bool]]          # nom -> (type, obligatoire)
    enums: dict[str, list[Any]] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    sensitive: bool = False                    # exige une confirmation explicite
    latency_s: tuple[float, float] = (0.02, 0.08)
    kind: str = "action"                       # "action" | "lecture"

    def compact_doc(self) -> str:
        parts = []
        for a, (t, req) in self.args.items():
            s = f"{a}:{t}" + ("" if req else "?")
            if a in self.enums:
                s += "{" + "|".join(map(str, self.enums[a])) + "}"
            parts.append(s)
        return f"{self.name}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Maison
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    entity_id: str
    name: str
    area: str
    domain: str
    state: dict[str, Any]


def default_home() -> list[Entity]:
    return [
        Entity("light.salon_plafond", "plafonnier du salon", "salon", "light", {"on": False, "brightness": 0}),
        Entity("light.salon_lampe", "lampe du salon", "salon", "light", {"on": True, "brightness": 40}),
        Entity("light.cuisine", "lumière de la cuisine", "cuisine", "light", {"on": False, "brightness": 0}),
        Entity("light.chambre", "lumière de la chambre", "chambre", "light", {"on": False, "brightness": 0}),
        Entity("cover.salon_volet", "volet du salon", "salon", "cover", {"position": 100}),
        Entity("cover.chambre_volet", "volet de la chambre", "chambre", "cover", {"position": 0}),
        Entity("climate.salon", "thermostat du salon", "salon", "climate", {"target": 19.0, "current": 18.4}),
        Entity("media_player.salon_tv", "télé du salon", "salon", "media_player", {"on": False, "volume": 20}),
        Entity("lock.porte_entree", "porte d'entrée", "entree", "lock", {"locked": True}),
        Entity("alarm.maison", "alarme", "maison", "alarm", {"armed": False}),
        Entity("cover.garage", "porte du garage", "garage", "cover", {"position": 0}),
    ]


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class HomeSimulator:
    """Exécute des appels d'outils contre un état de maison simulé.

    ``fault_rate`` injecte des pannes (appareil injoignable, service externe lent) pour que
    l'étudiant apprenne à les annoncer sans inventer de résultat.
    """

    def __init__(self, entities: list[Entity] | None = None, seed: int = 0, fault_rate: float = 0.0):
        self.entities = {e.entity_id: e for e in (entities or default_home())}
        self.rng = random.Random(seed)
        self.fault_rate = fault_rate
        self.log: list[dict[str, Any]] = []
        self.tools: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self._register_defaults()

    # -- registre ------------------------------------------------------------

    def register(self, spec: ToolSpec, handler: Callable[[dict[str, Any]], Any]) -> None:
        self.tools[spec.name] = spec
        self._handlers[spec.name] = handler

    def _register_defaults(self) -> None:
        ids = list(self.entities)
        lights = [i for i in ids if i.startswith("light.")]
        covers = [i for i in ids if i.startswith("cover.")]
        self.register(ToolSpec("maison.allumer", "Allume une lumière ou un appareil",
                               {"entite": ("str", True)}, {"entite": lights + ["media_player.salon_tv"]}),
                      self._turn_on)
        self.register(ToolSpec("maison.eteindre", "Éteint une lumière ou un appareil",
                               {"entite": ("str", True)}, {"entite": lights + ["media_player.salon_tv"]}),
                      self._turn_off)
        self.register(ToolSpec("maison.luminosite", "Règle la luminosité en %",
                               {"entite": ("str", True), "pourcent": ("int", True)},
                               {"entite": lights}, {"pourcent": (0, 100)}),
                      self._brightness)
        self.register(ToolSpec("maison.volet", "Règle l'ouverture d'un volet ou d'une porte en %",
                               {"entite": ("str", True), "position": ("int", True)},
                               {"entite": covers}, {"position": (0, 100)}),
                      self._cover)
        self.register(ToolSpec("maison.thermostat", "Règle la consigne de température",
                               {"entite": ("str", True), "temperature": ("num", True)},
                               {"entite": ["climate.salon"]}, {"temperature": (7, 28)}),
                      self._climate)
        self.register(ToolSpec("maison.etat", "Lit l'état d'une entité",
                               {"entite": ("str", True)}, {"entite": ids}, kind="lecture"),
                      self._state)
        self.register(ToolSpec("maison.serrure", "Verrouille ou déverrouille une serrure",
                               {"entite": ("str", True), "verrouiller": ("bool", True)},
                               {"entite": ["lock.porte_entree"]}, sensitive=True),
                      self._lock)
        self.register(ToolSpec("maison.alarme", "Arme ou désarme l'alarme",
                               {"armer": ("bool", True)}, sensitive=True),
                      self._alarm)
        self.register(ToolSpec("meteo.prevision", "Prévision météo",
                               {"ville": ("str", True), "jour": ("str", True)},
                               {"jour": ["aujourdhui", "demain", "apres_demain"]},
                               latency_s=(0.6, 2.5), kind="lecture"),
                      self._weather)
        self.register(ToolSpec("web.recherche", "Recherche sur le web, renvoie des extraits",
                               {"requete": ("str", True)}, latency_s=(1.0, 4.0), kind="lecture"),
                      self._search)

    def tools_block(self) -> str:
        return " ; ".join(t.compact_doc() for t in self.tools.values())

    def entities_block(self) -> str:
        return " ; ".join(f"{e.entity_id}|{e.name}|{e.area}" for e in self.entities.values())

    # -- validation ----------------------------------------------------------

    def validate(self, call: ToolCall, confirmed: bool = False) -> None:
        spec = self.tools.get(call.name)
        if spec is None:
            raise ToolError("outil_inconnu", f"outil inconnu : {call.name}")
        for a, (t, req) in spec.args.items():
            if req and a not in call.args:
                raise ToolError("argument_manquant", f"argument manquant : {a}")
        for a, v in call.args.items():
            if a not in spec.args:
                raise ToolError("argument_inconnu", f"argument inconnu : {a}")
            t = spec.args[a][0]
            if t == "int" and isinstance(v, bool) or not isinstance(v, ARG_TYPES[t]):
                raise ToolError("type_invalide", f"{a} doit être de type {t}")
            if a in spec.enums and v not in spec.enums[a]:
                raise ToolError("valeur_invalide", f"{a}={v!r} hors des valeurs permises")
            if a in spec.ranges:
                lo, hi = spec.ranges[a]
                if not lo <= v <= hi:
                    raise ToolError("hors_bornes", f"{a}={v} hors de [{lo}, {hi}]")
        touches_sensitive = call.args.get("entite") in SENSITIVE_ENTITIES
        if (spec.sensitive or touches_sensitive) and not confirmed:
            raise ToolError("confirmation_requise", "action sensible : confirmation de l'utilisateur requise")

    # -- exécution -----------------------------------------------------------

    def execute(self, call: ToolCall, confirmed: bool = False) -> tuple[ToolResult, float]:
        """Renvoie le résultat et la latence simulée (secondes)."""
        spec = self.tools.get(call.name)
        latency = self.rng.uniform(*(spec.latency_s if spec else (0.01, 0.02)))
        entry = {"call": call.to_json(), "confirmed": confirmed}
        try:
            self.validate(call, confirmed)
            if self.fault_rate and self.rng.random() < self.fault_rate:
                raise ToolError("injoignable", "l'appareil ou le service ne répond pas")
            payload = self._handlers[call.name](call.args)
            res = ToolResult(call.call_id, "ok", payload)
        except ToolError as e:
            status = "refus" if e.code == "confirmation_requise" else "erreur"
            res = ToolResult(call.call_id, status, {"code": e.code, "message": str(e)})
        entry["result"] = {"status": res.status, "payload": res.payload}
        self.log.append(entry)
        return res, latency

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {k: copy.deepcopy(e.state) for k, e in self.entities.items()}

    # -- gestionnaires -------------------------------------------------------

    def _e(self, eid: str) -> Entity:
        return self.entities[eid]

    def _turn_on(self, a):
        e = self._e(a["entite"])
        e.state["on"] = True
        if "brightness" in e.state and e.state["brightness"] == 0:
            e.state["brightness"] = 100
        return {"entite": e.entity_id, "etat": "allume"}

    def _turn_off(self, a):
        e = self._e(a["entite"])
        e.state["on"] = False
        return {"entite": e.entity_id, "etat": "eteint"}

    def _brightness(self, a):
        e = self._e(a["entite"])
        e.state.update(on=a["pourcent"] > 0, brightness=a["pourcent"])
        return {"entite": e.entity_id, "luminosite": a["pourcent"]}

    def _cover(self, a):
        e = self._e(a["entite"])
        e.state["position"] = a["position"]
        return {"entite": e.entity_id, "position": a["position"]}

    def _climate(self, a):
        e = self._e(a["entite"])
        e.state["target"] = float(a["temperature"])
        return {"entite": e.entity_id, "consigne": float(a["temperature"]), "actuelle": e.state["current"]}

    def _state(self, a):
        e = self._e(a["entite"])
        return {"entite": e.entity_id, "nom": e.name, **e.state}

    def _lock(self, a):
        e = self._e(a["entite"])
        e.state["locked"] = a["verrouiller"]
        return {"entite": e.entity_id, "verrouillee": a["verrouiller"]}

    def _alarm(self, a):
        e = self._e("alarm.maison")
        e.state["armed"] = a["armer"]
        return {"armee": a["armer"]}

    def _weather(self, a):
        # crc32 et non hash() : hash() change d'un processus à l'autre, la météo doit être stable.
        r = random.Random(zlib.crc32(f"{a['ville'].lower()}|{a['jour']}".encode()))
        ciel = r.choice(["soleil", "nuageux", "pluie", "averses", "brouillard"])
        tmin = r.randint(-2, 14)
        return {"ville": a["ville"], "jour": a["jour"], "ciel": ciel,
                "temp_min": tmin, "temp_max": tmin + r.randint(3, 12),
                "pluie_mm": 0 if ciel in ("soleil", "nuageux") else r.randint(1, 15)}

    def _search(self, a):
        return {"requete": a["requete"], "extraits": [],
                "note": "simulateur : brancher un vrai moteur de recherche pour l'usine à données"}
