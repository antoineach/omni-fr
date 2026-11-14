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
sudo apt-get install -y -q build-essential cmake git wget curl ffmpeg sox libsox-fmt-all python3-venv python3-dev \
  libportaudio2 libasound2-plugins pulseaudio-utils

# Boîte à outils CUDA 12.8 (compilateur nvcc) depuis le dépôt NVIDIA, si absente.
# Pas le paquet Ubuntu « nvidia-cuda-toolkit » : CUDA 12.0, incompatible avec le GCC 13 d'Ubuntu 24.04.
export PATH=/usr/local/cuda/bin:$PATH
if ! command -v nvcc >/dev/null; then
  if grep -qi microsoft /proc/version; then distro=wsl-ubuntu; else distro=ubuntu2404; fi
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update -q
  sudo apt-get install -y -q cuda-toolkit-12-8
  grep -q /usr/local/cuda/bin ~/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
fi
nvcc --version | tail -1

# Rust (cargo) : la normalisation de texte de ZONOS3 est écrite en Rust
if ! command -v cargo >/dev/null && [ ! -x ~/.cargo/bin/cargo ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"

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

bash scripts/installer_voix.sh
python -m pytest -q
omnifr materiel
omnifr installer
omnifr telecharger
echo
echo "Prêt. Activer l'environnement : . .venv/bin/activate"
echo "  omnifr tester --micro-test   # vérifie toute la chaîne, WAV dans ~/omnifr-modeles/test"
echo "  omnifr discuter      # au clavier"
echo "  omnifr assistant     # à la voix (« Hey Jarvis »)"
