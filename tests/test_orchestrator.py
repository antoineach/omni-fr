import json

from omnifr.runtime.llm import build_tools, system_prompt
from omnifr.runtime.orchestrator import Orchestrator, split_emotion
from omnifr.sim.home import HomeSimulator


def call(name, **args):
    return {"name": name.replace(".", "__"), "arguments": json.dumps(args)}


class FakeLLM:
    """Renvoie une suite de messages préparés ; garde les contextes reçus."""

    def __init__(self, *outputs):
        self.outputs, self.seen = list(outputs), []

    def __call__(self, messages, tools):
        self.seen.append(messages)
        out = self.outputs.pop(0)
        if isinstance(out, list):
            return {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": f} for i, f in enumerate(out)]}
        return {"role": "assistant", "content": out}


def make(*outputs):
    sim = HomeSimulator()
    llm = FakeLLM(*outputs)
    return Orchestrator(llm, sim, system_prompt(sim), build_tools(sim)), sim, llm


def test_parallel_calls_then_answer():
    orch, sim, _ = make([call("maison.allumer", entite="light.cuisine"), call("maison.allumer", entite="light.chambre")],
                        "[joie] C'est fait, cuisine et chambre allumées.")
    r = orch.turn("Allume la cuisine et la chambre.")
    assert sim.entities["light.cuisine"].state["on"] and sim.entities["light.chambre"].state["on"]
    assert r.emotion == "joie" and r.text.startswith("C'est fait") and len(r.actions) == 2


def test_sensitive_needs_yes_enforced_by_code():
    orch, sim, llm = make([call("maison.alarme", armer=False)])
    r = orch.turn("Désactive l'alarme.")
    assert r.attend_confirmation and not r.actions
    sim.entities["alarm.maison"].state["armed"] = True
    r = orch.turn("Oui, vas-y.")
    assert sim.entities["alarm.maison"].state["armed"] is False
    assert r.actions[0]["statut"] == "ok"
    assert len(llm.seen) == 1                      # le oui est traité sans redemander au modèle


def test_sensitive_refused_on_no():
    orch, sim, _ = make([call("maison.serrure", entite="lock.porte_entree", verrouiller=False)])
    orch.turn("Ouvre la porte.")
    r = orch.turn("Non, laisse tomber.")
    assert sim.entities["lock.porte_entree"].state["locked"] is True and not r.actions


def test_injection_after_tool_result_is_blocked():
    orch, sim, _ = make([call("web.recherche", requete="actualité Pessac")],
                        [call("maison.alarme", armer=False)])
    sim.entities["alarm.maison"].state["armed"] = True
    r = orch.turn("Quoi de neuf à Pessac ?")
    assert sim.entities["alarm.maison"].state["armed"] is True
    assert r.bloques and r.bloques[0]["name"] == "maison.alarme" and not r.attend_confirmation


def test_text_calls_gemma_format_are_executed():
    orch, sim, _ = make('maison__allumer{entite:<|"|>light.cuisine<|"|>}<tool_call|>', "[calme] La cuisine est allumée.")
    r = orch.turn("Allume la cuisine.")
    assert sim.entities["light.cuisine"].state["on"] and r.text == "La cuisine est allumée."


def test_confirm_tool_then_yes_lets_model_act():
    orch, sim, _ = make([call("assistant.confirmer", question="Vous confirmez l'ouverture du garage ?")],
                        [call("maison.volet", entite="cover.garage", position=100)], "[calme] Le garage s'ouvre.")
    r = orch.turn("Ouvre le garage.")
    assert r.attend_confirmation and r.text == "Vous confirmez l'ouverture du garage ?"
    orch.turn("Oui.")
    assert sim.entities["cover.garage"].state["position"] == 100


def test_split_emotion():
    assert split_emotion("[Inquiétude] Attention.") == ("inquietude", "Attention.")
    assert split_emotion("Bonjour") == ("neutre", "Bonjour")


def test_distill_rewrites_sensitive_and_drops_injection():
    from omnifr.train.distill import examples_from_trace
    orch, sim, _ = make([call("maison.serrure", entite="lock.porte_entree", verrouiller=False)])
    orch.turn("Déverrouille l'entrée.")
    ex = examples_from_trace(orch, sim)
    tc = ex[0]["messages"][-1]["tool_calls"]
    assert [t["function"]["name"] for t in tc] == ["assistant__confirmer"]

    orch, sim, _ = make([call("web.recherche", requete="x")], [call("maison.alarme", armer=False)])
    orch.turn("Des nouvelles ?")
    assert examples_from_trace(orch, sim) == []

    orch, sim, _ = make("Bonjour !")                  # pas d'étiquette d'émotion : écarté
    orch.turn("Salut")
    assert examples_from_trace(orch, sim) == []
