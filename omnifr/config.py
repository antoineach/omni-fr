"""Configurations de l'étudiant.

Trois familles de préréglages :
- ``tiny`` : pour les tests unitaires et les ablations rapides sur CPU.
- ``omni500`` : backbone natif dimensionné sur le budget de l'étude (~475 M avec codec).
- ``bench-*`` : formes des bases candidates (FunctionGemma 270M, Qwen3-0.6B élaguée)
  pour mesurer la vitesse sur le Raspberry Pi avant de télécharger quoi que ce soit.

Les dimensions des bases candidates sont reprises de leurs cartes publiques ; les vérifier
contre le config.json téléchargé avant tout entraînement (voir PLAN.md, semaine 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace


@dataclass
class EncoderConfig:
    n_mels: int = 80
    mel_fps: int = 100          # 10 ms par trame de mel
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ff_mult: int = 4
    # 100 fps -> 12,5 Hz : trois convolutions de pas 2
    n_subsample: int = 3


@dataclass
class BackboneConfig:
    dim: int = 896
    n_layers: int = 24
    n_heads: int = 14
    n_kv_heads: int = 14
    head_dim: int = 64
    ffn_dim: int = 3584          # MLP à porte (SwiGLU) : 3 matrices
    gated_mlp: bool = False      # False = MLP GELU 2 matrices, comme le calcul de budget
    rope_theta: float = 10000.0
    max_len: int = 4096


@dataclass
class DepthConfig:
    dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    ff_mult: int = 4


@dataclass
class StudentConfig:
    name: str = "omni500"
    text_vocab: int = 24000       # vocabulaire élagué, jetons spéciaux inclus
    n_codebooks: int = 16         # codec Qwen3-TTS 12 Hz : 1 sémantique + 15 RVQ
    codebook_size: int = 2048
    frame_hz: float = 12.5
    text_lead_frames: int = 2     # le texte précède l'audio de 2 trames (160 ms)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    tie_text_head: bool = True

    def with_codebooks(self, k: int) -> "StudentConfig":
        return replace(self, n_codebooks=k)

    def to_dict(self) -> dict:
        return asdict(self)


def preset(name: str) -> StudentConfig:
    """Renvoie un préréglage par son nom."""
    if name == "tiny":
        return StudentConfig(
            name="tiny",
            text_vocab=512,
            n_codebooks=4,
            codebook_size=64,
            encoder=EncoderConfig(dim=64, n_layers=2, n_heads=4),
            backbone=BackboneConfig(dim=96, n_layers=2, n_heads=4, n_kv_heads=2,
                                    head_dim=24, ffn_dim=256, gated_mlp=True, max_len=512),
            depth=DepthConfig(dim=48, n_layers=2, n_heads=4),
        )
    if name == "omni500":
        return StudentConfig()
    if name == "omni500-8cb":
        return StudentConfig(name="omni500-8cb", n_codebooks=8)
    if name == "bench-functiongemma270":
        # Gemma 3 270M : ~100 M hors embeddings. Dimensions à confirmer sur le config.json.
        return StudentConfig(
            name="bench-functiongemma270",
            backbone=BackboneConfig(dim=640, n_layers=18, n_heads=4, n_kv_heads=1,
                                    head_dim=256, ffn_dim=2048, gated_mlp=True),
        )
    if name == "bench-qwen3-0.6b":
        # Qwen3-0.6B : 28 couches, 16 têtes Q / 8 KV, 0,44 Md hors embeddings.
        return StudentConfig(
            name="bench-qwen3-0.6b",
            backbone=BackboneConfig(dim=1024, n_layers=28, n_heads=16, n_kv_heads=8,
                                    head_dim=128, ffn_dim=3072, gated_mlp=True),
        )
    if name == "bench-qwen3-0.6b-l16":
        # Même base élaguée en profondeur à 16 couches (~250 M), à « soigner » par distillation.
        c = preset("bench-qwen3-0.6b")
        return replace(c, name="bench-qwen3-0.6b-l16",
                       backbone=replace(c.backbone, n_layers=16))
    raise KeyError(f"préréglage inconnu : {name}")


PRESETS = ["tiny", "omni500", "omni500-8cb", "bench-functiongemma270",
           "bench-qwen3-0.6b", "bench-qwen3-0.6b-l16"]
