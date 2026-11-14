"""Serveur local Qwen3-TTS en flux, lancé dans son propre environnement Python (.venv-voix).

Moteur : faster-qwen3-tts (graphes CUDA + génération en flux). Le premier morceau d'audio part
au bout de quelques centaines de ms au lieu d'attendre la phrase entière. Ce fichier n'importe
rien d'Omni-FR.

    .venv-voix/bin/python omnifr/runtime/qwen_tts_server.py --port 8095 --mode expressif
    POST /tts {"text": "...", "emotion": "joie"}
      -> flux PCM float32 mono (en-tête X-Sample-Rate), morceau par morceau
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
    ap.add_argument("--backend", choices=["torch", "ggml"], default="torch",
                    help="ggml : moteur C++ qwentts.cpp (pip install 'faster-qwen3-tts[ggml]'), expérimental")
    ap.add_argument("--morceau", type=int, default=8,
                    help="pas de codec par morceau (12 Hz : 8 = 667 ms). Plus gros = meilleur débit, premier son plus tardif")
    a = ap.parse_args()

    import threading

    import numpy as np
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    kind = "Base" if a.mode == "clone" else "CustomVoice"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16   # Turing (2070) : float16
    if a.backend == "ggml":
        m = FasterQwen3TTS.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-{a.taille}-{kind}", backend="ggml")
    else:
        m = FasterQwen3TTS.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-{a.taille}-{kind}", device="cuda", dtype=dtype)
    print(f"moteur : {a.backend}", flush=True)
    speaker = a.nom
    if a.mode == "expressif":
        inner = getattr(m, "model", None)
        names = []
        for attr in ("get_supported_speakers", "get_supported_spks"):
            if inner is not None and hasattr(inner, attr):
                names = list(getattr(inner, attr)())
                break
        speaker = speaker or ("ryan" if not names or "ryan" in names else names[0])
        print(f"voix disponibles : {', '.join(names) or '?'} ; choisie : {speaker}", flush=True)
    m.warmup(prefill_len=100)          # capture des graphes CUDA une fois pour toutes
    lock = threading.Lock()            # un seul flux à la fois sur le GPU

    def stream(text: str, emotion: str):
        instr = INSTRUCT.get(emotion, "") or None
        if a.mode == "clone":
            return m.generate_voice_clone_streaming(text=text, language="French", ref_audio=a.voix,
                                                    ref_text=a.texte_voix, xvec_only=not a.texte_voix,
                                                    chunk_size=a.morceau)
        return m.generate_custom_voice_streaming(text=text, speaker=speaker, language="French", instruct=instr,
                                                 chunk_size=a.morceau)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                gen = stream(req["text"], req.get("emotion", "neutre"))
                try:
                    first = next(gen)
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}".encode()
                    self.send_response(500)
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("X-Sample-Rate", str(first[1]))
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    for item in _chain(first, gen):
                        data = np.asarray(item[0], dtype=np.float32).reshape(-1).tobytes()
                        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass             # l'utilisateur a coupé la parole : le client a fermé la connexion

    print(f"prêt sur le port {a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


def _chain(first, gen):
    yield first
    yield from gen


if __name__ == "__main__":
    main()
