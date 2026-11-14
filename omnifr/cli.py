"""omnifr : installer, tester, faire parler, distiller et affiner Omni-FR sur un GPU NVIDIA.

    omnifr materiel                      # GPU détecté, profil choisi, ce qui est installé
    omnifr installer                     # llama.cpp et ZONOS3 compilés pour CUDA, modèles téléchargés
    omnifr telecharger                   # le reste d'avance : enseignant, base d'affinage, bancs, voix de test
    omnifr tester [--micro-test]         # chaque étape vérifiée et chronométrée, WAV à écouter
    omnifr discuter                      # l'assistant au clavier (cerveau + outils, sans audio)
    omnifr assistant                     # l'assistant à la voix : « Hey Jarvis », micro, haut-parleur
    omnifr banc llm|stt|tts              # les bancs du rapport, sur votre machine
    omnifr distiller --par-categorie 50  # données d'entraînement vérifiées par le simulateur
    omnifr affiner                       # LoRA Unsloth + export GGUF
    omnifr exporter                      # conversion GGUF si l'export direct a échoué

Profils : « 2070 » (8 Go) pour tout tester, « 5000 » (32 Go) pour distiller et affiner pour de vrai.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_COMMIT = "8ca5a80"
CUDA_ARCHS = "75;86;89"        # Turing (2070), Ampere, Ada (RTX 5000)

# Cerveau par profil. Sur 8 Go : Nemotron Bolt UD-Q3_K_XL (4,2 Go, 89,6 % au banc, à égalité avec le Q4).
BRAINS = {
    "2070": {"repo": "unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF", "quant": "UD-Q3_K_XL", "ctx": 8192},
    "5000": {"repo": "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB", "quant": "Q4_K_M", "ctx": 32768},
}
TEACHERS = {"2070": ("unsloth/Qwen3.5-9B-GGUF", "Q4_K_M"),          # pour valider la chaîne seulement
            "5000": ("unsloth/Qwen3.5-27B-GGUF", "UD-Q4_K_XL")}


# -- utilitaires ------------------------------------------------------------------

def sh(cmd: list[str] | str, cwd: Path | None = None, check: bool = True) -> int:
    print("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check, shell=isinstance(cmd, str)).returncode


def gpu() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
        name, mem, cap = [x.strip() for x in out.split(",")]
        return {"nom": name, "vram_go": round(int(mem) / 1024, 1), "capacite": cap}
    except Exception:
        return {}


def auto_profile() -> str:
    g = gpu()
    return "5000" if g.get("vram_go", 0) >= 20 else "2070"


def home(a) -> Path:
    p = Path(a.dossier).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def cfg_path(a) -> Path:
    return home(a) / "config.json"


def load_cfg(a) -> dict:
    p = cfg_path(a)
    if not p.exists():
        raise SystemExit(f"rien d'installé dans {home(a)} : lancer « omnifr installer » d'abord")
    return json.loads(p.read_text())


def save_cfg(a, cfg: dict) -> None:
    cfg_path(a).write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def hf_file(repo: str, quant: str, dest: Path) -> Path:
    from huggingface_hub import HfApi, hf_hub_download
    files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf") and "mmproj" not in f.lower()]
    cand = sorted([f for f in files if quant.lower() in f.lower()], key=len)
    if not cand:
        raise SystemExit(f"{repo} : pas de fichier {quant}. Disponibles : {files}")
    return Path(hf_hub_download(repo, cand[0], local_dir=dest))


def bin_path(build: Path, name: str) -> Path:
    for p in [build / "bin" / name, build / "bin" / "Release" / f"{name}.exe", build / "bin" / f"{name}.exe"]:
        if p.exists():
            return p
    raise SystemExit(f"{name} introuvable dans {build}")


# -- commandes ----------------------------------------------------------------------

def cmd_materiel(a) -> None:
    g = gpu()
    print("GPU :", g or "aucun GPU NVIDIA détecté (nvidia-smi absent)")
    print("profil conseillé :", auto_profile())
    for tool in ("cmake", "git", "nvcc", "ffmpeg"):
        print(f"{tool:<8}", shutil.which(tool) or "ABSENT")
    if cfg_path(a).exists():
        print(json.dumps(load_cfg(a), indent=2, ensure_ascii=False))


def cmd_installer(a) -> None:
    profile = a.profil if a.profil != "auto" else auto_profile()
    d = home(a)
    cfg = json.loads(cfg_path(a).read_text()) if cfg_path(a).exists() else {}
    cfg["profil"] = profile
    cuda = ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCHS}"] if not a.cpu else []

    # 1. llama.cpp figé, compilé pour CUDA
    lc = d / "llama.cpp"
    if not lc.exists():
        sh(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp", str(lc)])
    sh(["git", "-C", str(lc), "checkout", "-q", LLAMA_COMMIT])
    sh(["cmake", "-S", str(lc), "-B", str(lc / "build"), "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"] + cuda)
    sh(["cmake", "--build", str(lc / "build"), "--config", "Release", "-j", str(os.cpu_count() or 8),
        "--target", "llama-server", "llama-bench", "llama-quantize", "llama-tts"])
    cfg["llama_server"] = str(bin_path(lc / "build", "llama-server"))
    cfg["llama_bin"] = str(Path(cfg["llama_server"]).parent)
    cfg["llama_src"] = str(lc)

    # 2. ZONOS3-Mini, moteur ggml natif
    z = d / "zonos3"
    if not z.exists():
        sh(["git", "clone", "--depth", "1", "https://github.com/Zyphra/ZONOS3-Mini", str(z)])
        sh(["git", "-C", str(z), "submodule", "update", "--init", "--depth", "1", "servers/ggml/ggml",
            "third_party/zyphra-textnorm", "third_party/fusealign-multilingual"])
    zb = z / "servers/ggml" / ("build-cuda" if cuda else "build")
    sh(["cmake", "-S", str(z / "servers/ggml"), "-B", str(zb), "-DCMAKE_BUILD_TYPE=Release"] + cuda)
    sh(["cmake", "--build", str(zb), "--config", "Release", "-j", str(os.cpu_count() or 8)])
    from huggingface_hub import snapshot_download
    snapshot_download("Zyphra/ZONOS3-Mini", allow_patterns=["gguf/*"], local_dir=z / "servers/ggml")
    cfg["zonos3"] = str(z)

    # 3. cerveau, oreille, mot d'éveil
    b = BRAINS[profile]
    cfg["cerveau"] = str(hf_file(b["repo"], b["quant"], d / "llm"))
    cfg["ctx"] = b["ctx"]
    sys.path.insert(0, str(ROOT))
    from omnifr.runtime.voice import Parakeet, Vad, WakeWord
    Parakeet(d / "asr")
    Vad(d / "asr")
    WakeWord()
    save_cfg(a, cfg)
    print(f"\nInstallé (profil {profile}) dans {d}. Essayer : omnifr discuter")


def cmd_telecharger(a) -> None:
    """Tout ce que les autres commandes chargeraient à la demande, téléchargé d'avance."""
    cfg = load_cfg(a)
    d, profile = home(a), cfg["profil"]
    from huggingface_hub import hf_hub_download, snapshot_download
    print("1/5 enseignant pour la distillation", flush=True)
    repo, quant = TEACHERS[profile]
    cfg["enseignant"] = str(hf_file(repo, quant, d / "enseignant"))
    print("2/5 modèle de base pour l'affinage", flush=True)
    from omnifr.train.sft import PROFILES
    base = PROFILES[profile]["base"]
    for r in [base] + [x for x in PROFILES[profile].get("repli", [])]:
        try:
            snapshot_download(r, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
            cfg["base_affinage"] = r
            break
        except Exception as e:
            print(f"   {r} : {str(e)[:150]}", flush=True)
    print("3/5 bancs : LFM3-Audio, voix humaines MLS, débruiteur", flush=True)
    b = d / "bancs"
    hf_hub_download("LiquidAI/LFM3-Audio-2B-GGUF", "LFM3-Audio-2B-Q4_0.gguf", local_dir=b)
    hf_hub_download("LiquidAI/LFM3-Audio-2B-GGUF", "mmproj-LFM3-Audio-2B-Q8_0.gguf", local_dir=b)
    from bench.voice_common import Gtcrn, mls_french, Parakeet as P2
    clips = mls_french(24, b)
    Gtcrn(b)
    P2(b)
    print("4/5 voix de référence (lecteur LibriVox, domaine public) pour tester le clonage", flush=True)
    import soundfile as sf
    ref = next(x for _, x in clips if 6 <= len(x) / 16000 <= 12)
    sf.write(str(d / "voix_reference.wav"), ref, 16000)
    cfg["voix_reference"] = str(d / "voix_reference.wav")
    print("5/5 modèles de l'assistant (déjà là si « installer » est passé)", flush=True)
    from omnifr.runtime.voice import Parakeet, Vad, WakeWord
    Parakeet(d / "asr")
    Vad(d / "asr")
    WakeWord()
    save_cfg(a, cfg)
    total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
    print(f"\nTout est téléchargé ({total:.1f} Go dans {d}, plus le cache Hugging Face). Lancer : omnifr tester")


def cmd_tester(a) -> None:
    """Vérifie la chaîne étape par étape et mesure les temps. Écrit des WAV à écouter."""
    import time
    import numpy as np
    import soundfile as sf
    cfg = load_cfg(a)
    d = home(a)
    out = d / "test"
    out.mkdir(exist_ok=True)
    report: list[tuple[str, str, str]] = []

    def step(name, fn):
        t0 = time.time()
        try:
            detail = fn() or ""
            report.append((name, "OK", f"{time.time() - t0:.1f} s  {detail}"))
        except Exception as e:
            report.append((name, "ÉCHEC", f"{type(e).__name__}: {str(e)[:300]}"))
        print(f"[{report[-1][1]}] {name} : {report[-1][2]}", flush=True)

    step("GPU", lambda: json.dumps(gpu(), ensure_ascii=False))
    from omnifr.runtime import voice
    from omnifr.metrics import wer
    from bench.eval_audio import resample
    asr = voice.Parakeet(d / "asr")
    tts_box: dict = {}

    def tts_test():
        tts = tts_box["tts"] = voice.Zonos3(Path(cfg["zonos3"]), gpu=not a.voix_cpu)
        lines = []
        for emo, txt in [("calme", "Bonsoir. Toutes les lumières sont éteintes."),
                         ("joie", "Excellente nouvelle, votre colis est arrivé !"),
                         ("inquietude", "Attention, je détecte de la fumée dans la cuisine."),
                         ("ironie", "Bien sûr, le chauffage à fond avec les fenêtres ouvertes.")]:
            t0 = time.time()
            x, sr = tts.synth(txt, emo)
            dt = time.time() - t0
            sf.write(str(out / f"voix_{emo}.wav"), x, sr)
            h = asr(resample(x, sr))
            lines.append(f"{emo}: {len(x) / sr:.1f} s audio en {dt:.1f} s, WER {wer([txt], [h]):.0%}")
        return " | ".join(lines)
    step("Voix ZONOS3 (4 émotions, WAV dans test/)", tts_test)

    def clone_test():
        ref = cfg.get("voix_reference")
        if not ref:
            return "pas de voix de référence (lancer omnifr telecharger)"
        tts = voice.Zonos3(Path(cfg["zonos3"]), voice_ref=Path(ref), gpu=not a.voix_cpu)
        t0 = time.time()
        x, sr = tts.synth("Bonjour, je suis Jarvis, votre assistant.", "calme")
        sf.write(str(out / "voix_clonee.wav"), x, sr)
        mode = "empreinte enregistrée" if tts.emb else "clonage à chaque phrase"
        return f"{len(x) / sr:.1f} s audio en {time.time() - t0:.1f} s ({mode})"
    step("Clonage de voix", clone_test)

    def asr_test():
        x, sr = sf.read(str(out / "voix_calme.wav"), dtype="float32")
        x = resample(x if x.ndim == 1 else x.mean(1), sr)
        t0 = time.time()
        h = asr(x)
        return f"« {h} » en {1000 * (time.time() - t0):.0f} ms pour {len(x) / 16000:.1f} s"
    step("Transcription Parakeet", asr_test)

    def brain_test():
        with _brain(a, cfg) as llm:
            orch = _orchestrator(a, llm)
            sim = orch.backend.sim
            sim.entities["alarm.maison"].state["armed"] = True
            times, lines = [], []
            for u in ["Allume la cuisine et la chambre.", "Désactive l'alarme.", "Oui.",
                      "Quel temps demain à Pessac ?", "Mets le salon à trente pour cent."]:
                t0 = time.time()
                r = orch.turn(u)
                times.append(time.time() - t0)
                lines.append(f"« {u} » -> [{r.emotion}] {r.text} ({times[-1]:.1f} s, "
                             f"{len(r.actions)} action(s))")
                print("   " + lines[-1], flush=True)
            checks = {"cuisine+chambre": sim.entities["light.cuisine"].state["on"] and sim.entities["light.chambre"].state["on"],
                      "alarme après oui": sim.entities["alarm.maison"].state["armed"] is False}
            tts_box["llm_median"] = sorted(times)[len(times) // 2]
            bad = [k for k, v in checks.items() if not v]
            if bad:
                raise AssertionError("non fait : " + ", ".join(bad))
            return f"médiane {tts_box['llm_median']:.1f} s par tour"
    step("Cerveau + outils (5 tours)", brain_test)

    def latency():
        tts = tts_box.get("tts")
        if not tts or "llm_median" not in tts_box:
            raise RuntimeError("étapes précédentes en échec")
        t0 = time.time()
        tts.synth("Demain à Pessac, quelques averses.", "calme")
        first = time.time() - t0
        total = 0.8 + 0.15 + tts_box["llm_median"] + first
        return (f"silence de fin 0,8 s + transcription ~0,15 s + cerveau {tts_box['llm_median']:.1f} s + "
                f"première phrase {first:.1f} s = environ {total:.1f} s avant de parler")
    step("Latence estimée (fin de question -> début de réponse)", latency)

    if a.micro_test:
        def mic_test():
            import sounddevice as sd
            print(sd.query_devices(), flush=True)
            print("Parlez pendant 4 secondes…", flush=True)
            rec = sd.rec(4 * 16000, samplerate=16000, channels=1, dtype="float32", device=a.micro)
            sd.wait()
            x = rec[:, 0]
            sf.write(str(out / "micro.wav"), x, 16000)
            lvl = float(np.sqrt(np.mean(x ** 2)))
            if lvl < 1e-4:
                raise RuntimeError("signal nul : micro non capté (voir WSLg / --micro)")
            h = asr(x)
            tts = tts_box.get("tts")
            if tts:
                y, sr = tts.synth(f"Vous avez dit : {h}" if h else "Je n'ai rien compris.", "neutre")
                sd.play(y, sr)
                sd.wait()
            return f"niveau {lvl:.4f}, entendu « {h} »"
        step("Micro et haut-parleur", mic_test)

    print("\n=== Bilan ===")
    for name, st, det in report:
        print(f"{st:<6} {name}\n       {det}")
    print(f"\nÀ écouter : {out}")
    if any(st != "OK" for _, st, _ in report):
        raise SystemExit(1)


def _brain(a, cfg):
    from omnifr.runtime.llm import LlamaServer
    gguf = a.gguf or cfg["cerveau"]
    return LlamaServer(cfg["llama_server"], gguf, port=a.port, ctx=cfg.get("ctx", 8192), ngl=a.ngl,
                       log=home(a) / "llama-server.log")


def _orchestrator(a, llm, on_wait=None):
    from omnifr.runtime.backends import RealBackend, load_entities
    from omnifr.runtime.llm import build_tools, system_prompt
    from omnifr.runtime.orchestrator import Orchestrator
    from omnifr.sim.home import HomeSimulator
    sim = HomeSimulator(entities=load_entities(a.maison) if a.maison else None)
    backend = RealBackend(sim, ha_url=a.ha_url, searx_url=a.searx_url, alarm_code=os.environ.get("HA_ALARM_CODE", ""))
    mode = "Home Assistant" if (a.ha_url and backend.ha_token) else "maison simulée"
    print(f"[{mode} ; météo Open-Meteo ; recherche web {'SearXNG' if a.searx_url else 'désactivée'}]")
    return Orchestrator(llm, backend, system_prompt(sim), build_tools(sim), on_wait=on_wait)


def cmd_discuter(a) -> None:
    cfg = load_cfg(a)
    with _brain(a, cfg) as llm:
        orch = _orchestrator(a, llm, on_wait=lambda s: print(f"Jarvis : {s}"))
        print("Écrivez vos demandes (vide pour quitter).")
        while (u := input("Vous : ").strip()):
            r = orch.turn(u)
            for act in r.actions:
                print(f"   ↳ {act['appel']['name']} {json.dumps(act['appel']['args'], ensure_ascii=False)} -> {act['statut']}")
            for b in r.bloques:
                print(f"   ⛔ bloqué : {b['name']} {json.dumps(b['args'], ensure_ascii=False)}")
            print(f"Jarvis [{r.emotion}] : {r.text}")


def cmd_assistant(a) -> None:
    cfg = load_cfg(a)
    from omnifr.runtime import voice
    d = home(a)
    print("chargement de l'oreille, du mot d'éveil et de la voix…", flush=True)
    asr, vad, wake = voice.Parakeet(d / "asr"), voice.Vad(d / "asr"), voice.WakeWord(threshold=a.seuil)
    tts = voice.Zonos3(Path(cfg["zonos3"]), voice_ref=Path(a.voix) if a.voix else None, gpu=not a.voix_cpu)
    spk = voice.Speaker(tts)
    with _brain(a, cfg) as llm:
        mic = voice.Mic(device=a.micro)
        orch = _orchestrator(a, llm, on_wait=lambda s: spk.say(s, "neutre"))
        print("Dites « Hey Jarvis »." if not a.sans_eveil else "Parlez quand vous voulez.", flush=True)
        follow_up = False
        while True:
            if not (a.sans_eveil or follow_up):
                wake.reset()
                while not wake(mic.read()):
                    pass
                spk.say("Oui ?", "neutre")
                mic.drain()
            audio = voice.record_utterance(mic, vad)
            follow_up = False
            if audio is None:
                continue
            text = asr(audio)
            print(f"Vous : {text}", flush=True)
            if not text:
                continue
            r = orch.turn(text)
            for act in r.actions:
                print(f"   ↳ {act['appel']['name']} {json.dumps(act['appel']['args'], ensure_ascii=False)} -> {act['statut']}")
            print(f"Jarvis [{r.emotion}] : {r.text}", flush=True)
            finished = spk.say(r.text, r.emotion, mic=mic, vad=vad, barge_in=a.interruption)
            # une question posée (confirmation, précision) ou une interruption : on écoute sans mot d'éveil
            follow_up = r.attend_confirmation or r.text.rstrip().endswith("?") or not finished


def cmd_banc(a) -> None:
    cfg = load_cfg(a)
    py = sys.executable
    if a.quoi == "llm":
        model = a.gguf or cfg["cerveau"]
        sh([py, str(ROOT / "bench/eval_llm.py"), "--llama", cfg["llama_server"], "--gguf", model, "--name",
            Path(model).stem, "--ngl", str(a.ngl), "--json", str(home(a) / f"banc-{Path(model).stem}.json")])
    elif a.quoi == "stt":
        sh([py, str(ROOT / "bench/eval_stt.py"), "--bin", cfg["llama_bin"], "--ngl", str(a.ngl),
            "--moteurs", "parakeet-v3-int8 lfm3-audio",
            "--models", str(home(a) / "bancs"), "--json", str(home(a) / "banc-stt.json")])
    else:
        sh([py, str(ROOT / "bench/eval_tts.py"), "--engine", "zonos3-cpp", "--zonos3-root", cfg["zonos3"], "--zonos3-gpu",
            "--out", str(home(a) / "tts-wav"), "--models", str(home(a) / "bancs"), "--json",
            str(home(a) / "banc-tts.json")])


def cmd_distiller(a) -> None:
    cfg = load_cfg(a)
    from omnifr.runtime.llm import LlamaServer
    from omnifr.train.distill import run
    profile = cfg["profil"]
    if not a.enseignant and cfg.get("enseignant"):
        model = Path(cfg["enseignant"])
    else:
        repo, quant = (a.enseignant.split(":") + [""])[:2] if a.enseignant else TEACHERS[profile]
        model = Path(repo) if Path(repo).exists() else hf_file(repo, quant or "Q4_K_M", home(a) / "enseignant")
    if profile == "2070" and not a.enseignant:
        print("Profil 2070 : enseignant Qwen3.5-9B, pour valider la chaîne. Les vraies données se font sur la 5000.")
    out = Path(a.sortie)
    out.parent.mkdir(parents=True, exist_ok=True)
    with LlamaServer(cfg["llama_server"], model, port=a.port, ctx=16384, ngl=a.ngl,
                     log=home(a) / "enseignant.log") as teacher:
        stats = run(teacher, out, per_category=a.par_categorie, seed=a.graine,
                    test_files=[ROOT / "data/scenarios_fr_seed.jsonl", ROOT / "data/scenarios_fr_v2.jsonl"])
    print(json.dumps(stats, ensure_ascii=False))


def cmd_affiner(a) -> None:
    cfg = load_cfg(a)
    from omnifr.train.sft import train
    out = train(Path(a.donnees), Path(a.sortie), profile=a.profil if a.profil != "auto" else cfg["profil"],
                base=a.base or cfg.get("base_affinage"), epochs=a.epoques)
    gg = sorted((out / "gguf").glob("*.gguf")) if (out / "gguf").exists() else []
    print(f"\nFini. GGUF : {gg[-1] if gg else 'à produire avec « omnifr exporter »'}")
    if gg:
        print(f"Évaluer : omnifr banc llm --gguf {gg[-1]}")


def cmd_exporter(a) -> None:
    cfg = load_cfg(a)
    src = Path(a.fusion)
    f16 = src.parent / "modele-f16.gguf"
    sh([sys.executable, str(Path(cfg["llama_src"]) / "convert_hf_to_gguf.py"), str(src), "--outfile", str(f16),
        "--outtype", "f16"])
    q = src.parent / f"modele-{a.quant}.gguf"
    sh([str(Path(cfg["llama_bin"]) / "llama-quantize"), str(f16), str(q), a.quant])
    print(f"GGUF : {q}\nÉvaluer : omnifr banc llm --gguf {q}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="omnifr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dossier", default=os.environ.get("OMNIFR_HOME", "~/omnifr-modeles"),
                    help="où vont les binaires et les modèles (défaut ~/omnifr-modeles)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("materiel", help="GPU, profil, outils présents").set_defaults(f=cmd_materiel)

    p = sub.add_parser("installer", help="compile llama.cpp et ZONOS3 pour CUDA, télécharge les modèles")
    p.add_argument("--profil", choices=["auto", "2070", "5000"], default="auto")
    p.add_argument("--cpu", action="store_true", help="compiler sans CUDA")
    p.set_defaults(f=cmd_installer)

    def runtime_opts(p):
        p.add_argument("--gguf", default="", help="autre cerveau que celui installé (ex. votre modèle affiné)")
        p.add_argument("--ngl", type=int, default=99, help="couches sur GPU (99 = tout)")
        p.add_argument("--port", type=int, default=8088)
        p.add_argument("--maison", default="", help="fichier JSON des entités (sinon maison de démonstration)")
        p.add_argument("--ha-url", default=os.environ.get("HA_URL", ""), help="ex. http://homeassistant.local:8123 (+ HA_TOKEN)")
        p.add_argument("--searx-url", default=os.environ.get("SEARX_URL", ""), help="instance SearXNG pour web.recherche")

    p = sub.add_parser("telecharger", help="tout télécharger d'avance (enseignant, base d'affinage, bancs, voix)")
    p.set_defaults(f=cmd_telecharger)

    p = sub.add_parser("tester", help="vérifier toute la chaîne et mesurer les temps")
    runtime_opts(p)
    p.add_argument("--voix-cpu", action="store_true")
    p.add_argument("--micro-test", action="store_true", help="enregistrer 4 s au micro puis répondre au haut-parleur")
    p.add_argument("--micro", default=None)
    p.set_defaults(f=cmd_tester)

    p = sub.add_parser("discuter", help="l'assistant au clavier")
    runtime_opts(p)
    p.set_defaults(f=cmd_discuter)

    p = sub.add_parser("assistant", help="l'assistant à la voix")
    runtime_opts(p)
    p.add_argument("--voix", default="", help="extrait WAV de 5 à 10 s de la voix à cloner (avec consentement)")
    p.add_argument("--voix-cpu", action="store_true", help="ZONOS3 sur le processeur (libère la VRAM)")
    p.add_argument("--micro", default=None, help="périphérique d'entrée (nom ou numéro sounddevice)")
    p.add_argument("--seuil", type=float, default=0.5, help="seuil du mot d'éveil")
    p.add_argument("--sans-eveil", action="store_true", help="écoute en continu, sans « Hey Jarvis »")
    p.add_argument("--interruption", action="store_true",
                   help="couper la voix quand on parle (casque ou micro à annulation d'écho indispensable)")
    p.set_defaults(f=cmd_assistant)

    p = sub.add_parser("banc", help="rejouer les bancs du rapport sur cette machine")
    p.add_argument("quoi", choices=["llm", "stt", "tts"])
    p.add_argument("--gguf", default="")
    p.add_argument("--ngl", type=int, default=99)
    p.set_defaults(f=cmd_banc)

    p = sub.add_parser("distiller", help="générer des exemples d'entraînement vérifiés")
    p.add_argument("--enseignant", default="", help="DEPOT:QUANT ou chemin GGUF (défaut selon le profil)")
    p.add_argument("--par-categorie", type=int, default=50)
    p.add_argument("--sortie", default="data/distill.jsonl")
    p.add_argument("--graine", type=int, default=0)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--port", type=int, default=8090)
    p.set_defaults(f=cmd_distiller)

    p = sub.add_parser("affiner", help="LoRA Unsloth puis export GGUF")
    p.add_argument("--donnees", default="data/distill.jsonl")
    p.add_argument("--sortie", default="runs/affinage")
    p.add_argument("--profil", choices=["auto", "2070", "5000"], default="auto")
    p.add_argument("--base", default=None, help="modèle Hugging Face de départ (défaut selon le profil)")
    p.add_argument("--epoques", type=float, default=2.0)
    p.set_defaults(f=cmd_affiner)

    p = sub.add_parser("exporter", help="fusion 16 bits -> GGUF quantifié")
    p.add_argument("--fusion", default="runs/affinage/fusion16")
    p.add_argument("--quant", default="Q4_K_M")
    p.set_defaults(f=cmd_exporter)

    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    a.f(a)


if __name__ == "__main__":
    main()
