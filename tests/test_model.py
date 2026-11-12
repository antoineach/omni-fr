import torch

from omnifr.config import preset
from omnifr.model import IGNORE, Batch, HFBackbone, OmniStudent


def _frames_batch(cfg, S, speak_from=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    K, card, V = cfg.n_codebooks, cfg.codebook_size, cfg.text_vocab
    tok_out = torch.randint(0, V, (1, S), generator=g)
    tok_in = torch.cat([torch.zeros(1, 1, dtype=torch.long), tok_out[:, :-1]], 1)
    audio_out = torch.randint(0, card, (1, S, K), generator=g)
    if speak_from is not None:
        audio_out[:, :speak_from] = IGNORE
    prev = audio_out.clone()
    prev[prev == IGNORE] = card
    audio_in = torch.cat([torch.full((1, 1, K), card), prev[:, :-1]], 1)
    return Batch(
        tok_in=tok_in,
        pos_type=torch.zeros(1, S, dtype=torch.long),
        frame_idx=torch.arange(S)[None],
        audio_in=audio_in,
        mel=torch.randn(1, S * 8, cfg.encoder.n_mels, generator=g),
        tok_out=tok_out,
        audio_out=audio_out,
    )


def test_forward_backward_finite():
    torch.manual_seed(0)
    cfg = preset("tiny")
    m = OmniStudent(cfg)
    b = _frames_batch(cfg, 20, speak_from=8)
    out = m(b)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_receptive_field():
    m = OmniStudent(preset("tiny"))
    assert m.encoder.receptive_mel_frames == 15
    assert m.encoder.step_window == 24


def test_streaming_matches_parallel():
    """Le pas à pas (encodeur + backbone avec caches) doit reproduire le calcul hors ligne."""
    torch.manual_seed(0)
    cfg = preset("tiny")
    m = OmniStudent(cfg).eval()
    S = 12
    b = _frames_batch(cfg, S)
    with torch.no_grad():
        ref = m(b)["text_logits"][0]
    state = m.new_state()
    win = m.encoder.step_window
    for t in range(S):
        end = (t + 1) * 8
        mel_w = b.mel[:, max(0, end - win):end]
        logits, _ = m.frame_step(state, mel_w, b.tok_in[:, t], b.audio_in[:, t])
        assert torch.allclose(logits[0], ref[t], atol=1e-4), f"écart à la trame {t}"


def test_depth_generate_matches_teacher_forcing():
    torch.manual_seed(0)
    cfg = preset("tiny")
    m = OmniStudent(cfg).eval()
    h = torch.randn(3, m.backbone.dim)
    tok = torch.tensor([1, 2, 3])
    codes = m.depth.generate(h, tok)
    logits = m.depth(h, tok, codes)
    assert torch.equal(logits.argmax(-1), codes)


def test_ctrl_positions_do_not_need_audio():
    torch.manual_seed(0)
    cfg = preset("tiny")
    m = OmniStudent(cfg).eval()
    state = m.new_state()
    m.frame_step(state, torch.randn(1, 8, cfg.encoder.n_mels), torch.tensor([1]),
                 torch.full((1, cfg.n_codebooks), cfg.codebook_size))
    for tok in (5, 6, 7):
        logits, _ = m.ctrl_step(state, torch.tensor([tok]))
        assert torch.isfinite(logits).all()
    assert state["bb"][0].length == 4


def test_hf_backbone_parallel_vs_cache():
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    cfg = preset("tiny")
    hf = Qwen3Model(Qwen3Config(hidden_size=96, intermediate_size=192, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=2, head_dim=24,
                                vocab_size=16)).eval()
    m = OmniStudent(cfg, backbone=HFBackbone(hf)).eval()
    S = 6
    b = _frames_batch(cfg, S)
    with torch.no_grad():
        ref = m(b)["text_logits"][0]
    state = m.new_state()
    for t in range(S):
        end = (t + 1) * 8
        logits, _ = m.frame_step(state, b.mel[:, max(0, end - m.encoder.step_window):end], b.tok_in[:, t], b.audio_in[:, t])
        assert torch.allclose(logits[0], ref[t], atol=1e-4)
