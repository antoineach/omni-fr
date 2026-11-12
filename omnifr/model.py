"""Modèle étudiant d'Omni-FR.

Une seule séquence causale mélange deux types de positions (voir ``protocol.PosType``) :

- **FRAME** (80 ms) : l'entrée somme les traits de l'encodeur audio pour cette trame, le jeton
  texte précédent et les codes audio que l'assistant a produits à la trame précédente. La sortie
  prédit le jeton texte de la trame et, si l'assistant parle, ses codes audio via le Depth
  Transformer.
- **CTRL** (rafale) : seule entrée, le jeton précédent. Sert aux appels d'outils, émis sans faire
  avancer l'horloge audio.

Le backbone est interchangeable : ``NativeBackbone`` (dimensionnement libre) ou ``HFBackbone``
(n'importe quel LM Hugging Face, alimenté par ``inputs_embeds``). Avec une base Gemma, penser à
multiplier les embeddings initialisés depuis la base par sqrt(dim), comme le fait Gemma.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BackboneConfig, DepthConfig, EncoderConfig, StudentConfig

IGNORE = -100


# ---------------------------------------------------------------------------
# Briques Transformer avec cache KV
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x : (B, H, T, D) ; cos/sin : (T, D/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return out.flatten(-2).type_as(x)


class KVCache:
    def __init__(self) -> None:
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv, self.hd = n_heads, n_kv_heads, head_dim
        self.q = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x, cos, sin, cache: Optional[KVCache] = None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        past = 0 if cache is None else cache.length
        q = apply_rope(q, cos[past:past + T], sin[past:past + T])
        k = apply_rope(k, cos[past:past + T], sin[past:past + T])
        if cache is not None:
            k, v = cache.append(k, v)
        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        S = k.shape[2]
        if T == 1:
            y = F.scaled_dot_product_attention(q, k, v)
        elif past == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, gated: bool):
        super().__init__()
        self.gated = gated
        self.up = nn.Linear(dim, hidden, bias=False)
        self.gate = nn.Linear(dim, hidden, bias=False) if gated else None
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        if self.gated:
            return self.down(F.silu(self.gate(x)) * self.up(x))
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, dim, n_heads, n_kv, head_dim, ffn, gated):
        super().__init__()
        self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv, head_dim)
        self.mlp = MLP(dim, ffn, gated)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, cache)
        return x + self.mlp(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, dim, n_layers, n_heads, n_kv, head_dim, ffn, gated, max_len, theta=10000.0):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim, n_heads, n_kv, head_dim, ffn, gated) for _ in range(n_layers))
        self.norm = RMSNorm(dim)
        cos, sin = rope_tables(head_dim, max_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def new_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.blocks]

    def forward(self, x, cache: Optional[list[KVCache]] = None):
        for i, b in enumerate(self.blocks):
            x = b(x, self.cos, self.sin, None if cache is None else cache[i])
        return self.norm(x), cache


# ---------------------------------------------------------------------------
# Encodeur audio causal
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Conv1d):
    def __init__(self, cin, cout, k, stride):
        super().__init__(cin, cout, k, stride=stride)
        self.left = k - 1

    def forward(self, x):
        return super().forward(F.pad(x, (self.left, 0)))


class AudioEncoder(nn.Module):
    """Log-mel 100 fps -> traits 12,5 Hz, strictement causal.

    Placeholder entraînable depuis zéro ; le plan prévoit de l'initialiser depuis un encodeur
    pré-entraîné français (Pantagruel B, ~93 M) rendu causal.
    """

    def __init__(self, cfg: EncoderConfig, out_dim: int, max_len: int = 4096):
        super().__init__()
        self.cfg = cfg
        convs, c = [], cfg.n_mels
        for _ in range(cfg.n_subsample):
            convs += [CausalConv1d(c, cfg.dim, 3, 2), nn.GELU()]
            c = cfg.dim
        self.sub = nn.Sequential(*convs)
        self.stride = 2 ** cfg.n_subsample
        self.tf = Transformer(cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_heads, cfg.dim // cfg.n_heads,
                              cfg.dim * cfg.ff_mult, False, max_len)
        self.proj = nn.Linear(cfg.dim, out_dim)

    @property
    def receptive_mel_frames(self) -> int:
        rf = 1
        for _ in range(self.cfg.n_subsample):
            rf = rf * 2 + 1          # noyau 3, pas 2
        return rf

    @property
    def step_window(self) -> int:
        """Fenêtre de mels à passer à ``step`` : couvre le champ réceptif ET le remplissage des
        couches intermédiaires (vérifié par test : 24 pour 3 couches ; 16 ne suffit pas)."""
        return self.stride * self.cfg.n_subsample

    def forward(self, mel: torch.Tensor, cache=None) -> torch.Tensor:
        # mel : (B, T_mel, n_mels), T_mel multiple de stride
        x = self.sub(mel.transpose(1, 2)).transpose(1, 2)
        x, _ = self.tf(x, cache)
        return self.proj(x)

    def step(self, mel_window: torch.Tensor, cache: list[KVCache]) -> torch.Tensor:
        """Une trame de 12,5 Hz à partir des derniers ``window`` mels (multiple de stride).

        La fenêtre doit couvrir le champ réceptif des convolutions ; seule la dernière sortie
        est gardée, ce qui reproduit exactement le calcul hors ligne.
        """
        x = self.sub(mel_window.transpose(1, 2)).transpose(1, 2)[:, -1:]
        x, _ = self.tf(x, cache)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

class NativeBackbone(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.dim = cfg.dim
        self.tf = Transformer(cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                              cfg.ffn_dim, cfg.gated_mlp, cfg.max_len, cfg.rope_theta)

    def new_cache(self):
        return self.tf.new_cache()

    def forward(self, x, cache=None):
        return self.tf(x, cache)


class HFBackbone(nn.Module):
    """Enveloppe un modèle de base Hugging Face (sans tête LM) piloté par ``inputs_embeds``."""

    def __init__(self, hf_base: nn.Module):
        super().__init__()
        self.m = hf_base
        self.dim = hf_base.config.hidden_size

    def new_cache(self):
        from transformers import DynamicCache
        return [DynamicCache()]

    def forward(self, x, cache=None):
        pkv = None if cache is None else cache[0]
        out = self.m(inputs_embeds=x, past_key_values=pkv, use_cache=cache is not None)
        return out.last_hidden_state, cache


# ---------------------------------------------------------------------------
# Depth Transformer : codes audio d'une trame, codebook par codebook
# ---------------------------------------------------------------------------

class DepthTransformer(nn.Module):
    def __init__(self, cfg: DepthConfig, in_dim: int, n_codebooks: int, card: int, text_vocab: int):
        super().__init__()
        self.K, self.card = n_codebooks, card
        self.inp = nn.Linear(in_dim, cfg.dim, bias=False)
        self.text_emb = nn.Embedding(text_vocab, cfg.dim)
        self.code_emb = nn.ModuleList(nn.Embedding(card, cfg.dim) for _ in range(n_codebooks - 1))
        self.tf = Transformer(cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_heads, cfg.dim // cfg.n_heads,
                              cfg.dim * cfg.ff_mult, False, max_len=n_codebooks)
        self.heads = nn.ModuleList(nn.Linear(cfg.dim, card, bias=False) for _ in range(n_codebooks))

    def _inputs(self, h, text_tok, codes):
        # h : (N, D) ; text_tok : (N,) ; codes : (N, k) codes déjà connus
        base = self.inp(h) + self.text_emb(text_tok)
        xs = [base]
        for j in range(codes.shape[1]):
            xs.append(base + self.code_emb[j](codes[:, j]))
        return torch.stack(xs, dim=1)

    def forward(self, h, text_tok, codes):
        """Forçage par l'enseignant : renvoie les logits (N, K, card)."""
        x = self._inputs(h, text_tok, codes[:, : self.K - 1])
        y, _ = self.tf(x)
        return torch.stack([self.heads[k](y[:, k]) for k in range(self.K)], dim=1)

    @torch.no_grad()
    def generate(self, h, text_tok, temperature: float = 0.0, generator=None):
        """Génère les K codes d'une trame. Renvoie (N, K)."""
        cache = self.tf.new_cache()
        base = self.inp(h) + self.text_emb(text_tok)
        x = base[:, None]
        out = []
        for k in range(self.K):
            y, _ = self.tf(x, cache)
            logits = self.heads[k](y[:, -1])
            if temperature <= 0:
                c = logits.argmax(-1)
            else:
                c = torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=generator)[:, 0]
            out.append(c)
            if k < self.K - 1:
                x = (base + self.code_emb[k](c))[:, None]
        return torch.stack(out, dim=1)


# ---------------------------------------------------------------------------
# Étudiant complet
# ---------------------------------------------------------------------------

@dataclass
class Batch:
    """Une séquence mélangée FRAME / CTRL, alignée position par position.

    tok_in    (B, S)    jeton d'entrée (le jeton cible de la position précédente)
    pos_type  (B, S)    0 = FRAME, 1 = CTRL
    frame_idx (B, S)    indice de trame audio de l'utilisateur pour les FRAME, -1 sinon
    audio_in  (B, S, K) codes de l'assistant à la trame précédente ; ``card`` = aucun
    mel       (B, F*8, n_mels) log-mel de l'utilisateur
    tok_out   (B, S)    jeton cible, IGNORE pour ne pas l'apprendre
    audio_out (B, S, K) codes cibles quand l'assistant parle, IGNORE sinon
    """
    tok_in: torch.Tensor
    pos_type: torch.Tensor
    frame_idx: torch.Tensor
    audio_in: torch.Tensor
    mel: torch.Tensor
    tok_out: torch.Tensor
    audio_out: torch.Tensor


class OmniStudent(nn.Module):
    def __init__(self, cfg: StudentConfig, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone if backbone is not None else NativeBackbone(cfg.backbone)
        d = self.backbone.dim
        self.encoder = AudioEncoder(cfg.encoder, d)
        self.text_emb = nn.Embedding(cfg.text_vocab, d)
        self.pos_type_emb = nn.Embedding(2, d)
        # +1 : index ``card`` = l'assistant ne parlait pas à la trame précédente
        self.audio_emb = nn.ModuleList(nn.Embedding(cfg.codebook_size + 1, d)
                                       for _ in range(cfg.n_codebooks))
        self.text_head = nn.Linear(d, cfg.text_vocab, bias=False)
        if cfg.tie_text_head:
            self.text_head.weight = self.text_emb.weight
        self.depth = DepthTransformer(cfg.depth, d, cfg.n_codebooks, cfg.codebook_size, cfg.text_vocab)
        w = torch.ones(cfg.n_codebooks)
        w[0] = 2.0                                  # le codebook sémantique porte le sens
        self.register_buffer("audio_w", w / w.mean(), persistent=False)

    # -- entrées ------------------------------------------------------------

    def embed(self, tok_in, pos_type, user_feat, audio_in):
        """user_feat : (B, S, D) déjà rangés par position (zéros sur les CTRL)."""
        x = self.text_emb(tok_in) + self.pos_type_emb(pos_type) + user_feat
        is_frame = (pos_type == 0).unsqueeze(-1).type_as(x)
        a = sum(self.audio_emb[k](audio_in[..., k]) for k in range(self.cfg.n_codebooks))
        return x + a * is_frame

    def gather_user_feat(self, enc, frame_idx):
        B, S = frame_idx.shape
        idx = frame_idx.clamp(min=0)
        feat = torch.gather(enc, 1, idx.unsqueeze(-1).expand(B, S, enc.shape[-1]))
        return feat * (frame_idx >= 0).unsqueeze(-1).type_as(feat)

    # -- entraînement -------------------------------------------------------

    def forward(self, b: Batch) -> dict[str, torch.Tensor]:
        enc = self.encoder(b.mel)
        x = self.embed(b.tok_in, b.pos_type, self.gather_user_feat(enc, b.frame_idx), b.audio_in)
        h, _ = self.backbone(x)
        text_logits = self.text_head(h)
        loss_text = F.cross_entropy(text_logits.flatten(0, 1), b.tok_out.flatten(), ignore_index=IGNORE)

        speak = (b.audio_out[..., 0] != IGNORE)
        if speak.any():
            hs = h[speak]
            tok = b.tok_out[speak].clamp(min=0)
            codes = b.audio_out[speak]
            logits = self.depth(hs, tok, codes.clamp(min=0))
            ce = F.cross_entropy(logits.flatten(0, 1), codes.flatten(), ignore_index=IGNORE,
                                 reduction="none").view_as(codes).float()
            loss_audio = (ce * self.audio_w).mean()
        else:
            loss_audio = h.sum() * 0.0
        return {"loss": loss_text + loss_audio, "text": loss_text, "audio": loss_audio,
                "text_logits": text_logits}

    # -- inférence pas à pas ------------------------------------------------

    def new_state(self) -> dict:
        return {"bb": self.backbone.new_cache(), "enc": self.encoder.tf.new_cache()}

    @torch.no_grad()
    def frame_step(self, state, mel_window, prev_tok, prev_audio):
        """Une trame FRAME. Renvoie (logits texte (B, V), état caché (B, D))."""
        feat = self.encoder.step(mel_window, state["enc"])
        pt = torch.zeros_like(prev_tok)
        x = self.embed(prev_tok[:, None], pt[:, None], feat, prev_audio[:, None])
        h, _ = self.backbone(x, state["bb"])
        return self.text_head(h[:, -1]), h[:, -1]

    @torch.no_grad()
    def ctrl_step(self, state, prev_tok):
        """Une position CTRL (rafale d'appel d'outil)."""
        B = prev_tok.shape[0]
        d = self.backbone.dim
        pt = torch.ones_like(prev_tok)
        none = torch.full((B, 1, self.cfg.n_codebooks), self.cfg.codebook_size, dtype=torch.long,
                          device=prev_tok.device)
        x = self.embed(prev_tok[:, None], pt[:, None], torch.zeros(B, 1, d, device=prev_tok.device), none)
        h, _ = self.backbone(x, state["bb"])
        return self.text_head(h[:, -1]), h[:, -1]


def count_params(model: OmniStudent) -> dict[str, int]:
    def n(m):
        return sum(p.numel() for p in m.parameters())

    out = {
        "encodeur": n(model.encoder),
        "embeddings texte (liés à la tête)": model.text_emb.weight.numel(),
        "backbone": n(model.backbone),
        "embeddings codes audio": sum(e.weight.numel() for e in model.audio_emb),
        "depth transformer": n(model.depth),
        "divers": model.pos_type_emb.weight.numel(),
    }
    out["total (hors décodeur du codec)"] = n(model)
    return out
