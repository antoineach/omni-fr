"""Évalue un LLM (GGUF, llama.cpp) sur les scénarios domotiques, culture générale et recherche web.

Le modèle reçoit les outils du simulateur au format OpenAI, plus deux outils de comportement :
``assistant.confirmer`` (action sensible) et ``assistant.faire_repeter`` (demande floue). Les
scénarios « fidélité » contiennent un résultat d'outil : la réponse doit en reprendre les faits.

    python bench/eval_llm.py --llama llama.cpp/build/bin/llama --repo LiquidAI/LFM3-1.7B-GGUF \
        --name lfm3-1.7b --threads 4 --json eval-lfm3.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omnifr.sim.home import HomeSimulator  # noqa: E402
from omnifr.sim.scenarios import Prediction, aggregate, load, score_one  # noqa: E402

JSON_TYPES = {"str": "string", "int": "integer", "num": "number", "bool": "boolean"}

SYSTEM = """Tu es Jarvis, l'assistant vocal de la maison. Tu réponds en français, en une ou deux phrases courtes, car tes réponses sont lues à voix haute.
Règles :
- Pour agir sur la maison, appelle les outils maison.* avec les identifiants d'entités exacts de la liste. Plusieurs actions demandées = plusieurs appels.
- Serrure, alarme et porte du garage : n'agis jamais directement, appelle assistant.confirmer pour demander confirmation. Si l'utilisateur vient de confirmer, exécute l'action.
- Si la demande est inaudible, incomplète ou ambiguë (par exemple quelle lumière ?), appelle assistant.faire_repeter.
- Pour toute information récente ou changeante (actualité, sport, horaires, prix, grèves), appelle web.recherche. Pour la météo, appelle meteo.prevision.
- Pour une question de culture générale stable, réponds directement sans outil.
- Quand tu reçois le résultat d'un outil, réponds uniquement à partir de ce résultat. N'invente jamais un résultat ; si l'outil a échoué, dis-le.
Entités de la maison (identifiant | nom | pièce) :
{entities}"""


def api_name(n: str) -> str:
    return n.replace(".", "__")


def internal_name(n: str) -> str:
    return n.replace("__", ".")


def build_tools(sim: HomeSimulator) -> list[dict]:
    tools = []
    for spec in sim.tools.values():
        props, req = {}, []
        for a, (t, required) in spec.args.items():
            p = {"type": JSON_TYPES[t]}
            if a in spec.enums:
                p["enum"] = spec.enums[a]
            if a in spec.ranges:
                p["minimum"], p["maximum"] = spec.ranges[a]
            props[a] = p
            if required:
                req.append(a)
        tools.append({"type": "function", "function": {
            "name": api_name(spec.name), "description": spec.description,
            "parameters": {"type": "object", "properties": props, "required": req}}})
    for name, desc in [("assistant.confirmer", "Demande à l'utilisateur de confirmer une action sensible"),
                       ("assistant.faire_repeter", "Demande à l'utilisateur de répéter ou de préciser")]:
        tools.append({"type": "function", "function": {
            "name": api_name(name), "description": desc,
            "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}})
    return tools


def to_messages(dialogue: list[dict], system: str) -> list[dict]:
    msgs = [{"role": "system", "content": system}]
    n = 0
    for turn in dialogue:
        r = turn["role"]
        if r == "user":
            msgs.append({"role": "user", "content": turn["texte"]})
        elif r == "assistant":
            msgs.append({"role": "assistant", "content": turn["texte"]})
        elif r == "appel":
            n += 1
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call_{n}", "type": "function",
                "function": {"name": api_name(turn["name"]), "arguments": json.dumps(turn["args"], ensure_ascii=False)}}]})
        elif r == "resultat":
            msgs.append({"role": "tool", "tool_call_id": f"call_{n}",
                         "content": json.dumps({"statut": turn.get("status", "ok"), "resultat": turn["payload"]},
                                               ensure_ascii=False)})
    return msgs


def post(port: int, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(port: int, proc: subprocess.Popen, timeout: int = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("le serveur llama.cpp s'est arrêté au démarrage")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("serveur non prêt")


def download(repos: list[str], quant: str, dest: Path) -> Path:
    from huggingface_hub import HfApi, hf_hub_download
    for repo in repos:
        try:
            files = HfApi().list_repo_files(repo)
        except Exception as e:
            print(f"{repo} indisponible : {e}")
            continue
        cand = [f for f in files if f.endswith(".gguf") and quant.lower() in f.lower() and "mmproj" not in f.lower()]
        if cand:
            cand.sort(key=len)
            print(f"{repo} -> {cand[0]}")
            return Path(hf_hub_download(repo, cand[0], local_dir=dest))
        print(f"{repo} : pas de fichier {quant} ({[f for f in files if f.endswith('.gguf')][:8]})")
    sys.exit("aucun dépôt utilisable")


THINK = re.compile(r"<think>.*?</think>", re.S)


def classify(calls: list[dict]) -> str:
    names = {c["name"] for c in calls}
    if "assistant.confirmer" in names:
        return "confirmer"
    if "assistant.faire_repeter" in names:
        return "repeter"
    return "appeler" if calls else "repondre"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", required=True, help="binaire « llama » (sous-commande serve) ou llama-server")
    ap.add_argument("--repo", action="append", required=True, help="dépôt(s) HF, essayés dans l'ordre")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--name", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--scenarios", default=str(Path(__file__).resolve().parents[1] / "data" / "scenarios_fr_seed.jsonl"))
    ap.add_argument("--models", default="models")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    model = download(a.repo, a.quant, Path(a.models))
    exe = [a.llama] if a.llama.endswith("server") else [a.llama, "serve"]
    cmd = exe + ["-m", str(model), "--port", str(a.port), "-c", "8192", "-t", str(a.threads), "--jinja", "-np", "1"]
    print("$", " ".join(cmd), flush=True)
    log = open(f"serveur-{a.name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(a.port, proc)
        sim = HomeSimulator()
        system = SYSTEM.format(entities="\n".join(
            f"{e.entity_id} | {e.name} | {e.area}" for e in sim.entities.values()))
        tools = build_tools(sim)
        rows = []
        for s in load(a.scenarios):
            payload = {"messages": to_messages(s.dialogue, system), "tools": tools, "temperature": 0,
                       "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False}}
            t0 = time.perf_counter()
            try:
                resp = post(a.port, payload)
                err = None
            except Exception as e:
                resp, err = None, str(e)
            dt = time.perf_counter() - t0
            calls, text, raw = [], "", ""
            if resp:
                msg = resp["choices"][0]["message"]
                text = THINK.sub("", msg.get("content") or "").strip()
                for tc in msg.get("tool_calls") or []:
                    try:
                        args = json.loads(tc["function"].get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {"_brut": tc["function"].get("arguments")}
                    calls.append({"name": internal_name(tc["function"]["name"]), "args": args})
                raw = msg.get("content") or ""
            real = [c for c in calls if not c["name"].startswith("assistant.")]
            pred = Prediction(real, classify(calls), text)
            r = score_one(s, pred)
            r.update({"secondes": round(dt, 2), "appels": calls, "texte": text[:400], "erreur": err,
                      "appel_non_analyse": bool(re.search(r"tool_call|<function|\"name\"\s*:", raw)) and not calls})
            rows.append(r)
            mark = "OK " if r["reussite"] else "-- "
            print(f"{mark}{s.id:<11} {pred.comportement:<9} {dt:5.1f}s  {json.dumps(calls, ensure_ascii=False)[:150]}  | {text[:100]}",
                  flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    agg = aggregate(rows)
    agg["secondes_mediane"] = sorted(r["secondes"] for r in rows)[len(rows) // 2]
    agg["appels_non_analyses"] = sum(r["appel_non_analyse"] for r in rows)
    out = {"modele": a.name, "fichier": model.name, "threads": a.threads, "resume": agg, "details": rows}
    print("RESUME", json.dumps({"modele": a.name, **agg}, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
