#!/usr/bin/env bash
# Installation complète sur un PC à GPU NVIDIA, sous Linux ou WSL2 (Ubuntu 24.04).
#   bash scripts/installer_gpu.sh            # tout : dépendances, CLI, compilation CUDA, modèles
#   bash scripts/installer_gpu.sh --sans-affinage
# Prérequis Windows : WSL2 + pilote NVIDIA Windows récent (le GPU est alors visible dans WSL,
# ne PAS installer de pilote NVIDIA dans Ubuntu). Micro et haut-parleur passent par WSLg (Windows 11).
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "nvidia-smi absent : pilote NVIDIA ?"; exit 1; }

sudo apt-get update -q
sudo apt-get install -y -q build-essential cmake git ffmpeg python3-venv python3-dev \
  libportaudio2 libasound2-plugins pulseaudio-utils

# Boîte à outils CUDA (compilateur nvcc), si absente
if ! command -v nvcc >/dev/null; then
  sudo apt-get install -y -q nvidia-cuda-toolkit || {
    echo "Installer CUDA 12.x : https://developer.nvidia.com/cuda-downloads (choisir WSL-Ubuntu sous WSL)"; exit 1; }
fi

# Sous WSL : faire passer ALSA par le serveur son de WSLg (micro et haut-parleur Windows)
if grep -qi microsoft /proc/version && [ ! -f ~/.asoundrc ]; then
  printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' > ~/.asoundrc
fi

python3 -m venv .venv
. .venv/bin/activate
pip install -U pip wheel
pip install -e ".[voix,dev]"
pip install --no-deps openwakeword==0.6.0
if [ "${1:-}" != "--sans-affinage" ]; then
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install -e ".[affinage,bancs]"
fi

python -m pytest -q
omnifr materiel
omnifr installer
echo
echo "Prêt. Activer l'environnement : . .venv/bin/activate"
echo "  omnifr discuter      # au clavier"
echo "  omnifr assistant     # à la voix (« Hey Jarvis »)"
