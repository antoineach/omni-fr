"""Serveur local Qwen3-TTS, lancé dans son propre environnement Python (.venv-voix).

qwen-tts impose transformers==4.57.3, incompatible avec le reste (Unsloth, Nemotron ASR en
transformers 5) : la voix tourne donc dans un processus séparé, avec son environnement à elle.
Ce fichier n'importe rien d'Omni-FR.

    .venv-voix/bin/python omnifr/runtime/qwen_tts_server.py --port 8095 --mode expressif
    POST /tts  {"text": "...", "emotion": "joie"}  ->  audio/wav
"""

from __future__ import annotations

import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

INSTRUCT = {"calme": "Speak in a calm, soft and reassuring tone.",
            "joie": "Speak in a cheerful, happy and lively tone.",
            "inquietude": "Speak in a worried, urgent and tense tone.",
            "ironie": "Speak in a dry, ironic, slightly amused tone, like a witty British butler.",
            "neutre": ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--mode", choices=["expressif", "clone"], default="expressif")
    ap.add_argument("--taille", default="0.6B")
    ap.add_argument("--voix", default="", help="extrait WAV (mode clone)")
    ap.add_argument("--texte-voix", default="", help="transcription de l'extrait (améliore le clonage)")
    ap.add_argument("--nom", default="", help="voix prédéfinie (mode expressif)")
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    kind = "Base" if a.mode == "clone" else "CustomVoice"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16   # Turing (2070) : float16
    m = Qwen3TTSModel.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-{a.taille}-{kind}", device_map="cuda:0",
                                      dtype=dtype, attn_implementation="sdpa")
    prompt, speaker = None, a.nom
    if a.mode == "clone":
        prompt = m.create_voice_clone_prompt(ref_audio=a.voix, ref_text=a.texte_voix or None,
                                             x_vector_only_mode=not a.texte_voix)
    else:
        names = []
        for attr in ("get_supported_speakers", "get_supported_spks"):
            if hasattr(m, attr):
                names = list(getattr(m, attr)())
                break
        speaker = speaker or ("Ryan" if not names or "Ryan" in names else names[0])
        print(f"voix disponibles : {', '.join(names) or '?'} ; choisie : {speaker}", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if a.mode == "clone":
                    wavs, sr = m.generate_voice_clone(text=req["text"], language="French", voice_clone_prompt=prompt)
                else:
                    wavs, sr = m.generate_custom_voice(text=req["text"], language="French", speaker=speaker,
                                                       instruct=INSTRUCT.get(req.get("emotion", "neutre"), ""))
                buf = io.BytesIO()
                sf.write(buf, np.asarray(wavs[0], dtype=np.float32).reshape(-1), sr, format="WAV")
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                self.wfile.write(buf.getvalue())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"{type(e).__name__}: {e}".encode())

    print(f"prêt sur le port {a.port}", flush=True)
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
