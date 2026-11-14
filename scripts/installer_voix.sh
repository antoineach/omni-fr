#!/usr/bin/env bash
# Environnement séparé pour Qwen3-TTS : qwen-tts impose transformers==4.57.3, incompatible avec
# Unsloth et Nemotron ASR. La voix tourne donc dans son propre processus, avec .venv-voix.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v sox >/dev/null || sudo apt-get install -y -q sox libsox-fmt-all
python3 -m venv .venv-voix
.venv-voix/bin/pip install -U pip wheel
.venv-voix/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# faster-qwen3-tts : graphes CUDA + génération en flux (la bibliothèque officielle n'a pas de flux).
# Il apporte qwen-tts-hf ; l'officiel qwen-tts ne doit pas cohabiter avec lui.
.venv-voix/bin/pip uninstall -y qwen-tts 2>/dev/null || true
# transformers figé à la version minimale de qwen-tts-hf : en 5.23, le codec (Mimi) a changé et le
# chargement plante dans Qwen3TTSTokenizerV2Encoder (self.upsample = None).
.venv-voix/bin/pip install faster-qwen3-tts "transformers==5.15.1"
.venv-voix/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for r in ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-Tokenizer-12Hz"):
    snapshot_download(r)
import torch, faster_qwen3_tts
print("faster-qwen3-tts OK | torch", torch.__version__, "| CUDA", torch.cuda.is_available())
PY
echo "Voix prête : omnifr discuter --parler"
