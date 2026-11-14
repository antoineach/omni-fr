import time
from datetime import datetime

import pytest

from omnifr.protocol import ToolCall
from omnifr.runtime.assistant_tools import Timers, calcul, maintenant, register_extra
from omnifr.sim.home import HomeSimulator, ToolError


def test_calcul():
    assert calcul("15 % de 87") == pytest.approx(13.05)
    assert calcul("2,5 x 4") == 10
    assert calcul("sqrt(144) + 2^3") == 20
    assert calcul("(17 + 3) / 4") == 5
    with pytest.raises(ToolError):
        calcul("__import__('os')")
    with pytest.raises(ToolError):
        calcul("1/0")


def test_maintenant_en_francais():
    m = maintenant(datetime(2026, 11, 14, 9, 5).astimezone())
    assert m["heure"] == "9 h 05" and m["date"] == "samedi 14 novembre 2026"


def test_minuteur_sonne_et_se_liste():
    rang = []
    t = Timers(on_ring=rang.append)
    sim = HomeSimulator()
    register_extra(sim, t)
    r, _ = sim.execute(ToolCall(1, "minuteur.lancer", {"duree_s": 1, "nom": "Pâtes"}))
    assert r.status == "ok" and r.payload["nom"] == "pâtes"
    assert sim.execute(ToolCall(2, "minuteur.liste", {}))[0].payload["minuteurs"][0]["nom"] == "pâtes"
    time.sleep(1.3)
    assert rang == ["pâtes"]
    assert sim.execute(ToolCall(3, "minuteur.lancer", {"duree_s": 0}))[0].status == "erreur"   # hors bornes


def test_outils_visibles_par_le_cerveau():
    from omnifr.runtime.llm import build_tools, system_prompt
    sim = HomeSimulator()
    register_extra(sim, Timers())
    names = {t["function"]["name"] for t in build_tools(sim)}
    assert {"temps__maintenant", "minuteur__lancer", "calcul__evaluer", "meteo__prevision", "web__recherche"} <= names
    assert "temps.maintenant" in system_prompt(sim)
