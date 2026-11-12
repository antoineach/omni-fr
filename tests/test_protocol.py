import pytest

from omnifr.metrics import normalize_fr, wer
from omnifr.protocol import (CallSyntaxError, ToolCall, ToolResult, Turn, WordTiming,
                             align_text_to_frames, parse_call, render_history, truncate_to_played)


def test_call_roundtrip():
    c = ToolCall(3, "meteo.prevision", {"ville": "Saint-Émilion", "jour": "demain", "n": 2,
                                        "x": 1.5, "ok": True, "rien": None})
    assert parse_call(c.to_compact()) == c


def test_call_with_spaces_and_quotes():
    c = ToolCall(1, "web.recherche", {"requete": 'horaires "TBM" ligne A'})
    assert parse_call(c.to_compact()) == c


@pytest.mark.parametrize("bad", [
    "<appel 1> meteo.prevision ville=Bordeaux </appel>",       # chaîne sans guillemets
    "<appel 1> Meteo ville=\"x\" </appel>",                     # majuscule dans le nom
    "<appel> meteo.prevision </appel>",                          # pas d'identifiant
    '<appel 1> m.p a="x" a="y" </appel>',                        # argument en double
])
def test_call_rejects(bad):
    with pytest.raises(CallSyntaxError):
        parse_call(bad)


def test_compact_call_is_short():
    """Raison d'être de la syntaxe compacte : rester loin d'un appel JSON verbeux."""
    c = ToolCall(1, "maison.luminosite", {"entite": "light.salon_plafond", "pourcent": 30})
    import json
    assert len(c.to_compact()) < 0.85 * len(json.dumps(c.to_json(), ensure_ascii=False))


def test_truncate_to_played():
    words = [WordTiming("Demain", 0.0, 0.4), WordTiming("il", 0.4, 0.55),
             WordTiming("pleuvra", 0.55, 1.0), WordTiming("beaucoup", 1.0, 1.5)]
    assert truncate_to_played(words, 1.2) == ("Demain il pleuvra", True)
    assert truncate_to_played(words, 9.0) == ("Demain il pleuvra beaucoup", False)


def test_history_marks_interruption_and_results():
    turns = [Turn("user", "Quel temps demain ?"),
             Turn("assistant", "Demain il", calls=[ToolCall(1, "meteo.prevision", {"ville": "Bordeaux", "jour": "demain"})],
                  results=[ToolResult(1, "ok", {"ciel": "pluie"})], interrupted=True),
             Turn("user", "Non, à Lyon.")]
    h = render_history(turns)
    assert "<interrompu>" in h and "<resultat 1 <ok>>" in h and h.startswith("<bos>")


def test_alignment_lead_and_lag():
    tok = lambda w: [len(w.strip())]          # un jeton par mot
    words = [WordTiming("a", 0.40, 0.5), WordTiming("b", 0.42, 0.5), WordTiming("c", 0.44, 0.5)]
    seq, lag = align_text_to_frames(words, 20, tok, pad_id=0, lead_frames=2)
    assert seq[3:6] == [1, 1, 1]              # trame 5 avancée de 2 -> 3, puis glissement
    assert lag == 2


def test_wer():
    assert wer(["Allume la cuisine."], ["allume la cuisine"]) == 0.0
    assert wer(["allume la cuisine"], ["allume cuisine"]) == pytest.approx(1 / 3)
    assert normalize_fr("L’alarme, s'il te plaît !") == "l' alarme s' il te plaît"
