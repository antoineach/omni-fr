# Omni-FR

Assistant vocal domotique français, compact (~500 M), parole en entrée et en sortie, appels
d'outils en parallèle, interruptible. Ce dépôt contient la phase 1 : l'architecture de
l'étudiant, le protocole de jetons, le simulateur de maison, les bancs de vitesse et le test du
codec. Le plan pas à pas est dans [PLAN.md](PLAN.md).

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt                 # cœur, CPU suffit
pip install -r requirements-teachers.txt        # codec, enseignants (PC à GPU)
python -m pytest -q                             # 24 tests, quelques secondes
```

Sous Windows, `sox` (pour `qwen-tts`) demande le binaire SoX dans le PATH pour certaines
fonctions ; le codec 12 Hz n'en a pas besoin, l'avertissement à l'import est sans effet.

## Commandes de la semaine 1

```bash
# Vitesse de l'étudiant (à lancer sur la machine cible ou un proxy ARM)
python -m bench.bench_step --preset all --codebooks 8 --threads 4 --int8 --json bench.json

# Vitesse du décodeur du codec en streaming
python -m bench.bench_codec --threads 4 --chunk 4

# Le codec restitue-t-il le français et votre voix ?
python scripts/codec_roundtrip.py --audio mes_wav/ --out sorties/codec --device cuda
```

Proxys ARM gratuits : workflow `.github/workflows/bench-arm.yml` (Mac M1 et Linux arm64, dépôt
public), et une instance Oracle Cloud Ampere A1.

## Organisation

| Chemin | Contenu |
| --- | --- |
| `omnifr/config.py` | Préréglages : `tiny`, `omni500`, et les formes des bases candidates |
| `omnifr/model.py` | Étudiant : encodeur causal, backbone natif ou Hugging Face, Depth Transformer, inférence pas à pas |
| `omnifr/protocol.py` | Jetons spéciaux, appels compacts, historique texte, troncature à l'interruption, alignement texte/trames |
| `omnifr/sim/home.py` | Simulateur de maison : validation des appels, confirmations, pannes, latences |
| `omnifr/sim/scenarios.py` | Jeu de scénarios et métriques (F1 outil, arguments exacts, réussite) |
| `omnifr/codec.py` | Codec Qwen3-TTS 12 Hz, avec décodage tronqué à moins de 16 codebooks |
| `bench/` | Bancs de vitesse de l'étudiant et du codec |
| `scripts/codec_roundtrip.py` | Expérience 1 : WER et similarité de locuteur après codec |
| `data/scenarios_fr_seed.jsonl` | 24 scénarios d'amorce en français |

## Choix de conception à ne pas défaire sans raison

- **Appels en rafale.** À 12,5 jetons/s, un appel JSON de 40 jetons prendrait 3,2 s. Les appels
  passent par des positions `CTRL` qui n'avancent pas l'horloge audio, en syntaxe compacte.
- **Aucun résultat d'outil n'est écrit par un modèle.** Seul le simulateur (puis l'orchestrateur réel) en produit.
- **Historique en texte seul**, tronqué au dernier mot réellement joué en cas d'interruption.
