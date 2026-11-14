"""Exécution réelle des outils : Home Assistant, météo Open-Meteo, recherche SearXNG.

Le simulateur reste la référence des schémas et des règles (validation des arguments, actions
sensibles). ``RealBackend`` valide chaque appel avec lui, puis l'exécute pour de vrai :

- ``maison.*`` → API REST de Home Assistant si ``--ha-url`` et ``HA_TOKEN`` sont fournis,
  sinon le simulateur (démonstration sans maison) ;
- ``meteo.prevision`` → Open-Meteo (gratuit, sans clé) ;
- ``web.recherche`` → une instance SearXNG si ``--searx-url`` est fourni, sinon une erreur
  explicite (l'assistant doit dire qu'il ne peut pas chercher, pas inventer).

Les entités de la maison se décrivent dans un fichier JSON :
    [{"id": "light.salon", "nom": "lumière du salon", "piece": "salon"}, ...]
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from ..protocol import ToolCall, ToolResult
from ..sim.home import Entity, HomeSimulator, ToolError

_JOURS = {"aujourdhui": 0, "demain": 1, "apres_demain": 2}
_WMO = {0: "soleil", 1: "plutôt dégagé", 2: "partiellement nuageux", 3: "couvert", 45: "brouillard",
        48: "brouillard givrant", 51: "bruine", 53: "bruine", 55: "bruine forte", 61: "pluie faible",
        63: "pluie", 65: "forte pluie", 66: "pluie verglaçante", 67: "pluie verglaçante", 71: "neige faible",
        73: "neige", 75: "forte neige", 77: "grains de neige", 80: "averses", 81: "averses", 82: "fortes averses",
        85: "averses de neige", 86: "averses de neige", 95: "orage", 96: "orage avec grêle", 99: "orage avec grêle"}


def _get(url: str, headers: dict | None = None, timeout: float = 8) -> Any:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _post(url: str, body: dict, headers: dict, timeout: float = 8) -> Any:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={**headers, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


def load_entities(path: str | Path) -> list[Entity]:
    domains = {"light": "light", "cover": "cover", "climate": "climate", "media_player": "media_player",
               "lock": "lock", "alarm_control_panel": "alarm", "switch": "light"}
    out = []
    for e in json.loads(Path(path).read_text(encoding="utf-8")):
        dom = domains.get(e["id"].split(".")[0], "light")
        out.append(Entity(e["id"], e["nom"], e.get("piece", ""), dom, {}))
    return out


class RealBackend:
    def __init__(self, sim: HomeSimulator, ha_url: str = "", ha_token: str = "", searx_url: str = "",
                 alarm_code: str = ""):
        self.sim = sim
        self.tools, self.entities = sim.tools, sim.entities
        self.ha_url, self.ha_token = ha_url.rstrip("/"), ha_token or os.environ.get("HA_TOKEN", "")
        self.searx_url, self.alarm_code = searx_url.rstrip("/"), alarm_code
        self.log: list[dict] = []

    # -- Home Assistant ------------------------------------------------------

    def _ha(self, domain: str, service: str, data: dict) -> Any:
        return _post(f"{self.ha_url}/api/services/{domain}/{service}", data,
                     {"Authorization": f"Bearer {self.ha_token}"})

    def _ha_state(self, eid: str) -> dict:
        s = _get(f"{self.ha_url}/api/states/{eid}", {"Authorization": f"Bearer {self.ha_token}"})
        return {"entite": eid, "etat": s.get("state"), **{k: v for k, v in s.get("attributes", {}).items()
                                                          if k in ("brightness", "current_position", "temperature",
                                                                   "current_temperature", "friendly_name")}}

    def _maison(self, name: str, a: dict) -> Any:
        if not (self.ha_url and self.ha_token):
            raise LookupError("simulateur")
        eid = a.get("entite", "")
        if name == "maison.allumer":
            self._ha("homeassistant", "turn_on", {"entity_id": eid})
        elif name == "maison.eteindre":
            self._ha("homeassistant", "turn_off", {"entity_id": eid})
        elif name == "maison.luminosite":
            self._ha("light", "turn_on", {"entity_id": eid, "brightness_pct": a["pourcent"]})
        elif name == "maison.volet":
            self._ha("cover", "set_cover_position", {"entity_id": eid, "position": a["position"]})
        elif name == "maison.thermostat":
            self._ha("climate", "set_temperature", {"entity_id": eid, "temperature": a["temperature"]})
        elif name == "maison.serrure":
            self._ha("lock", "lock" if a["verrouiller"] else "unlock", {"entity_id": eid})
        elif name == "maison.alarme":
            alarm = next((e for e in self.entities if e.startswith("alarm")), "alarm_control_panel.maison")
            data = {"entity_id": alarm, **({"code": self.alarm_code} if self.alarm_code else {})}
            self._ha("alarm_control_panel", "alarm_arm_away" if a["armer"] else "alarm_disarm", data)
        elif name == "maison.etat":
            return self._ha_state(eid)
        return {"entite": eid, "fait": True}

    # -- services externes ---------------------------------------------------

    def _meteo(self, a: dict) -> dict:
        geo = _get("https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
            {"name": a["ville"], "count": 1, "language": "fr", "format": "json"}))
        if not geo.get("results"):
            raise ToolError("ville_inconnue", f"ville introuvable : {a['ville']}")
        g = geo["results"][0]
        f = _get("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
            "latitude": g["latitude"], "longitude": g["longitude"], "timezone": "auto", "forecast_days": 3,
            "daily": "weather_code,temperature_2m_min,temperature_2m_max,precipitation_sum,wind_speed_10m_max"}))
        i, d = _JOURS[a["jour"]], f["daily"]
        return {"ville": g["name"], "jour": a["jour"], "ciel": _WMO.get(d["weather_code"][i], "variable"),
                "temp_min": d["temperature_2m_min"][i], "temp_max": d["temperature_2m_max"][i],
                "pluie_mm": d["precipitation_sum"][i], "vent_max_kmh": d["wind_speed_10m_max"][i]}

    def _wikipedia(self, q: str) -> dict:
        """Recherche sans clé ni serveur : Wikipédia en français (faits, dates, personnes, lieux)."""
        api = "https://fr.wikipedia.org/w/api.php?"
        h = {"User-Agent": "omni-fr/0.2 (assistant vocal local)"}
        hits = _get(api + urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": q, "srlimit": 3,
                                                  "format": "json"}), h).get("query", {}).get("search", [])
        if not hits:
            return {"requete": q, "source": "Wikipédia", "extraits": []}
        titles = "|".join(x["title"] for x in hits)
        pages = _get(api + urllib.parse.urlencode({"action": "query", "prop": "extracts", "exintro": 1,
                                                   "explaintext": 1, "exsentences": 4, "titles": titles,
                                                   "format": "json", "redirects": 1}), h)["query"]["pages"]
        order = {x["title"]: i for i, x in enumerate(hits)}
        ex = sorted(pages.values(), key=lambda p: order.get(p.get("title"), 9))
        return {"requete": q, "source": "Wikipédia",
                "extraits": [f"{p.get('title')} : {p.get('extract', '')[:600]}" for p in ex if p.get("extract")]}

    def _web(self, a: dict) -> dict:
        if not self.searx_url:
            return self._wikipedia(a["requete"])
        r = _get(f"{self.searx_url}/search?" + urllib.parse.urlencode(
            {"q": a["requete"], "format": "json", "language": "fr"}), timeout=10)
        return {"requete": a["requete"], "extraits": [
            f"{x.get('title', '')} : {x.get('content', '')} ({x.get('url', '')})" for x in r.get("results", [])[:5]]}

    # -- interface commune avec le simulateur --------------------------------

    def execute(self, call: ToolCall, confirmed: bool = False) -> tuple[ToolResult, float]:
        try:
            self.sim.validate(call, confirmed)
            if call.name == "meteo.prevision":
                payload = self._meteo(call.args)
            elif call.name == "web.recherche":
                payload = self._web(call.args)
            elif not call.name.startswith("maison."):
                return self.sim.execute(call, confirmed)     # heure, minuteurs, calculs : exécutés localement
            else:
                try:
                    payload = self._maison(call.name, call.args)
                except LookupError:
                    return self.sim.execute(call, confirmed)
            res = ToolResult(call.call_id, "ok", payload)
        except ToolError as e:
            res = ToolResult(call.call_id, "refus" if e.code == "confirmation_requise" else "erreur",
                             {"code": e.code, "message": str(e)})
        except Exception as e:  # réseau, Home Assistant injoignable…
            res = ToolResult(call.call_id, "erreur", {"code": "injoignable", "message": str(e)[:200]})
        self.log.append({"call": call.to_json(), "confirmed": confirmed, "status": res.status})
        return res, 0.0
