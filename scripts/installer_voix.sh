#!/usr/bin/env bash
# Environnement séparé pour Qwen3-TTS : qwen-tts impose transformers==4.57.3, incompatible avec
# Unsloth et Nemotron ASR. La voix tourne donc dans son propre processus, avec .venv-voix.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv-voix
.venv-voix/bin/pip install -U pip wheel
.venv-voix/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv-voix/bin/pip install qwen-tts
.venv-voix/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for r in ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-Tokenizer-12Hz"):
    snapshot_download(r)
import torch, qwen_tts
print("qwen-tts", getattr(qwen_tts, "__version__", "?"), "| torch", torch.__version__, "| CUDA", torch.cuda.is_available())
PY
echo "Voix prête : omnifr discuter --parler"
