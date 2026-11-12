"""Accès au codec Qwen3-TTS 12 Hz (16 codebooks de 2 048, 12,5 trames/s).

Dépend du paquet ``qwen-tts`` (``requirements-teachers.txt``). API vérifiée sur qwen-tts 0.1.1 :
``Qwen3TTSTokenizer.from_pretrained`` puis ``encode`` / ``decode``.

Le décodeur officiel refuse moins de 16 codebooks. ``decode(codes, n_codebooks=8)`` reproduit
son calcul en ne sommant que les premiers quantificateurs RVQ, ce qui est exactement ce qu'un
étudiant à 8 codebooks produirait. À valider à l'oreille : le décodeur a été entraîné avec 16.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

CODEC_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


class QwenCodec:
    def __init__(self, model_id: str = CODEC_ID, device: str = "cpu", dtype: torch.dtype = torch.float32):
        from qwen_tts import Qwen3TTSTokenizer  # import tardif : dépendance optionnelle

        self.tok = Qwen3TTSTokenizer.from_pretrained(model_id, device_map=device, dtype=dtype)
        self.model = self.tok.model
        self.sr_out = int(self.model.get_output_sample_rate())
        self.n_codebooks = int(self.model.config.decoder_config.num_quantizers)

    def encode(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        """Renvoie les codes (T, 16) en int64 sur CPU."""
        enc = self.tok.encode(wav.astype(np.float32), sr=sr)
        return enc.audio_codes[0].long().cpu()

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor, n_codebooks: Optional[int] = None) -> tuple[np.ndarray, int]:
        """codes : (T, K). Si K < 16 ou ``n_codebooks`` donné, décodage tronqué."""
        k = n_codebooks or codes.shape[1]
        if k == self.n_codebooks:
            wavs, sr = self.tok.decode({"audio_codes": codes})
            return wavs[0], sr
        dec = self.model.decoder
        c = codes[:, :k].T.unsqueeze(0).to(next(dec.parameters()).device)   # (1, K, T)

        def forward_k(x):  # copie de Decoder.forward sans la vérification du nombre de codebooks
            hidden = dec.quantizer.decode(x)
            hidden = dec.pre_conv(hidden).transpose(1, 2)
            hidden = dec.pre_transformer(inputs_embeds=hidden).last_hidden_state.permute(0, 2, 1)
            for blocks in dec.upsample:
                for block in blocks:
                    hidden = block(hidden)
            wav = hidden
            for block in dec.decoder:
                wav = block(wav)
            return wav.clamp(min=-1, max=1)

        # Même découpage que Decoder.chunked_decode (blocs de 300 trames, 25 de contexte gauche).
        chunk, left = 300, 25
        wavs, start = [], 0
        while start < c.shape[-1]:
            end = min(start + chunk, c.shape[-1])
            ctx = left if start - left > 0 else start
            w = forward_k(c[..., start - ctx:end])
            wavs.append(w[..., ctx * dec.total_upsample:])
            start = end
        wav = torch.cat(wavs, dim=-1)
        return wav[0, 0].float().cpu().numpy(), self.sr_out
