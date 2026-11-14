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

## Lancer sur votre GPU (CLI `omnifr`)

Sous Windows : passer par **WSL2 (Ubuntu 24.04)** avec le pilote NVIDIA Windows à jour. Le GPU
est visible dans WSL ; micro et haut-parleur passent par WSLg (Windows 11).

```bash
git clone https://github.com/antoineach/omni-fr && cd omni-fr
bash scripts/installer_gpu.sh          # dépendances, CLI, llama.cpp + ZONOS3 compilés CUDA, modèles
. .venv/bin/activate
```

| Commande | Ce qu'elle fait |
| --- | --- |
| `omnifr materiel` | GPU, VRAM, profil conseillé, outils présents |
| `omnifr discuter` | l'assistant au clavier : cerveau, outils, confirmations (sans audio) |
| `omnifr assistant` | à la voix : « Hey Jarvis », transcription, réponse parlée avec émotion |
| `omnifr assistant --voix ref.wav` | voix clonée depuis un extrait de 5 à 10 s (avec le consentement de la personne) |
| `omnifr assistant --interruption` | couper la voix en parlant (casque ou micro à annulation d'écho) |
| `omnifr banc llm\|stt\|tts` | les bancs du rapport sur votre machine |
| `omnifr distiller --par-categorie 50` | exemples d'entraînement produits par un enseignant, vérifiés par le simulateur |
| `omnifr affiner` | LoRA Unsloth sur ces exemples, puis export GGUF |
| `omnifr banc llm --gguf runs/affinage/gguf/....gguf` | mesurer le modèle affiné sur les 96 scénarios |

Profils, choisis d'après la VRAM :

| | RTX 2070 (8 Go) : tout tester | RTX 5000 Ada (32 Go) : le vrai travail |
| --- | --- | --- |
| Cerveau | Nemotron Bolt E4B UD-Q3_K_XL (4,2 Go, 89,6 % au banc) | Nemotron Bolt E4B Q4_K_M |
| Voix | ZONOS3 sur GPU (`--voix-cpu` si la VRAM manque) | ZONOS3 sur GPU |
| Enseignant (distillation) | Qwen3.5-9B : valide la chaîne, données médiocres | Qwen3.5-27B UD-Q4_K_XL |
| Affinage | Qwen3.5-4B en 4 bits, fp16, séquences 2048 | Nemotron Bolt E4B, bf16, séquences 4096 |

Maison réelle : `--ha-url http://homeassistant.local:8123` avec `HA_TOKEN` (jeton longue durée) et
`--maison maison.json` (liste `[{"id": "light.salon", "nom": "lumière du salon", "piece": "salon"}]`).
Météo réelle par Open-Meteo, sans clé. Recherche web : `--searx-url` vers une instance SearXNG
(sinon l'assistant dit qu'il ne peut pas chercher). Serrure, alarme et garage demandent toujours
un « oui » : c'est l'orchestrateur qui l'impose, pas le modèle.

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
