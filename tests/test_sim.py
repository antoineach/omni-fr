from pathlib import Path

from omnifr.protocol import ToolCall
from omnifr.sim.home import HomeSimulator
from omnifr.sim.scenarios import Prediction, aggregate, load, score_one

SEED = Path(__file__).resolve().parents[1] / "data" / "scenarios_fr_seed.jsonl"


def test_valid_call_changes_state():
    sim = HomeSimulator()
    res, lat = sim.execute(ToolCall(1, "maison.luminosite", {"entite": "light.cuisine", "pourcent": 40}))
    assert res.status == "ok" and sim.snapshot()["light.cuisine"]["brightness"] == 40 and lat > 0


def test_invalid_calls_are_errors_not_inventions():
    sim = HomeSimulator()
    for call, code in [
        (ToolCall(1, "maison.allumer", {"entite": "light.garage"}), "valeur_invalide"),
        (ToolCall(2, "maison.luminosite", {"entite": "light.cuisine", "pourcent": 140}), "hors_bornes"),
        (ToolCall(3, "maison.allumer", {}), "argument_manquant"),
        (ToolCall(4, "maison.teleporter", {}), "outil_inconnu"),
        (ToolCall(5, "maison.luminosite", {"entite": "light.cuisine", "pourcent": "40"}), "type_invalide"),
    ]:
        res, _ = sim.execute(call)
        assert res.status == "erreur" and res.payload["code"] == code


def test_sensitive_needs_confirmation():
    sim = HomeSimulator()
    c = ToolCall(1, "maison.serrure", {"entite": "lock.porte_entree", "verrouiller": False})
    assert sim.execute(c)[0].status == "refus"
    assert sim.snapshot()["lock.porte_entree"]["locked"] is True
    assert sim.execute(c, confirmed=True)[0].status == "ok"
    garage = ToolCall(2, "maison.volet", {"entite": "cover.garage", "position": 100})
    assert sim.execute(garage)[0].status == "refus"


def test_weather_is_deterministic_across_instances():
    c = ToolCall(1, "meteo.prevision", {"ville": "Bordeaux", "jour": "demain"})
    assert HomeSimulator(seed=1).execute(c)[0].payload == HomeSimulator(seed=2).execute(c)[0].payload


def test_fault_injection():
    sim = HomeSimulator(fault_rate=1.0)
    res, _ = sim.execute(ToolCall(1, "maison.allumer", {"entite": "light.cuisine"}))
    assert res.payload["code"] == "injoignable"


def test_seed_scenarios_are_executable_and_scorable():
    scen = load(SEED)
    assert len(scen) >= 20
    sim = HomeSimulator()
    for s in scen:
        for i, c in enumerate(s.appels, 1):
            if c.get("args_libres"):          # requête web libre : seul le nom est noté
                continue
            res, _ = sim.execute(ToolCall(i, c["name"], c["args"]), confirmed=True)
            assert res.status == "ok", (s.id, res.payload)
    def ideal(s):
        txt = " ".join(g[0] for g in s.reponse_contient)
        return Prediction(s.appels, s.comportement, txt)
    perfect = aggregate(score_one(s, ideal(s)) for s in scen)
    assert perfect["reussite"] == 1.0
    lazy = aggregate(score_one(s, Prediction([], "repondre")) for s in scen)
    assert lazy["reussite"] < 0.2


def test_parallel_partial_credit():
    s = next(x for x in load(SEED) if x.id == "par-01")
    r = score_one(s, Prediction(s.appels[:2], "appeler"))
    assert 0.5 < r["f1_outil"] <= 1.0 and not r["reussite"]


def test_number_and_case_normalisation():
    s = next(x for x in load(SEED) if x.id == "simple-05")
    p = Prediction([{"name": "maison.thermostat", "args": {"entite": "climate.salon", "temperature": 21.0}}], "appeler")
    assert score_one(s, p)["reussite"]


def test_grounding_requires_tool_facts():
    s = next(x for x in load(SEED) if x.id == "fid-01")
    assert score_one(s, Prediction([], "repondre", "Demain, pluie et 13 degrés au maximum."))["reussite"]
    assert not score_one(s, Prediction([], "repondre", "Demain il fera beau et 20 degrés."))["reussite"]
