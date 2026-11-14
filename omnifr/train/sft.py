"""Affinage LoRA (Unsloth) sur les exemples distillés, puis export GGUF pour llama.cpp.

Chaque exemple = un contexte (système, historique, résultats d'outils) et la sortie attendue de
l'assistant (appels d'outils ou phrase avec étiquette d'émotion). La perte ne porte que sur cette
sortie. Le masquage est calculé à partir du gabarit de conversation du modèle lui-même : pas de
marqueurs codés en dur, donc valable pour Gemma 4, Nemotron Bolt ou Qwen3.5.

Profils :
- ``2070`` (8 Go, Turing) : fp16, séquences de 2048, pour valider la chaîne sur un petit modèle ;
- ``5000`` (32 Go, Ada) : bf16, séquences de 4096, pour le vrai modèle (E4B).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PROFILES = {
    "2070": {"base": "unsloth/Qwen3.5-4B", "repli": ["Qwen/Qwen3.5-4B"], "max_len": 2048, "batch": 1, "accum": 8, "r": 16, "bf16": False},
    "5000": {"base": "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16", "repli": ["google/gemma-4-E4B-it"], "max_len": 4096, "batch": 2, "accum": 8, "r": 32,
             "bf16": True},
}


def _text_tok(tok):
    return getattr(tok, "tokenizer", tok)


def encode(tok, ex: dict, max_len: int) -> dict | None:
    """input_ids et labels : seuls les jetons de la dernière réponse de l'assistant comptent."""
    t = _text_tok(tok)
    msgs, tools = ex["messages"], ex.get("tools")
    for m in msgs:                       # certains gabarits refusent un contenu None
        if m.get("content") is None:
            m["content"] = ""
        for tc in m.get("tool_calls") or []:
            if isinstance(tc["function"].get("arguments"), str):
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"] or "{}")
    try:
        prefix = t.apply_chat_template(msgs[:-1], tools=tools, add_generation_prompt=True, tokenize=False)
        full = t.apply_chat_template(msgs, tools=tools, tokenize=False)
    except Exception:
        return None
    if not full.startswith(prefix):
        return None
    p_ids = t(prefix, add_special_tokens=False)["input_ids"]
    f_ids = t(full, add_special_tokens=False)["input_ids"]
    if len(f_ids) > max_len or len(f_ids) <= len(p_ids):
        return None
    return {"input_ids": f_ids, "labels": [-100] * len(p_ids) + f_ids[len(p_ids):]}


def train(data: Path, out: Path, profile: str = "2070", base: str | None = None, epochs: float = 2.0,
          lr: float = 2e-4, seed: int = 0, val_frac: float = 0.05) -> Path:
    import torch
    from unsloth import FastModel          # importer unsloth avant transformers
    from transformers import Trainer, TrainingArguments

    P = PROFILES[profile]
    model, tok = FastModel.from_pretrained(base or P["base"], max_seq_length=P["max_len"], load_in_4bit=True,
                                           full_finetuning=False)
    model = FastModel.get_peft_model(model, r=P["r"], lora_alpha=P["r"], lora_dropout=0.0,
                                     finetune_vision_layers=False, finetune_language_layers=True,
                                     finetune_attention_modules=True, finetune_mlp_modules=True,
                                     use_gradient_checkpointing="unsloth", random_state=seed)

    rows = [json.loads(x) for x in Path(data).read_text(encoding="utf-8").splitlines() if x.strip()]
    enc = [e for e in (encode(tok, r, P["max_len"]) for r in rows) if e]
    print(f"{len(enc)} exemples encodés sur {len(rows)}", flush=True)
    if not enc:
        raise SystemExit("aucun exemple utilisable : vérifier le gabarit du modèle")
    random.Random(seed).shuffle(enc)
    n_val = max(1, int(len(enc) * val_frac))
    val, tr = enc[:n_val], enc[n_val:]
    pad = _text_tok(tok).pad_token_id or 0

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad)
        lab = torch.full((len(batch), L), -100)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n], lab[i, :n], att[i, :n] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    args = TrainingArguments(
        output_dir=str(out / "ckpt"), per_device_train_batch_size=P["batch"], gradient_accumulation_steps=P["accum"],
        num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        bf16=P["bf16"], fp16=not P["bf16"], optim="adamw_8bit", weight_decay=0.0, seed=seed,
        report_to="none", remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=tr, eval_dataset=val, data_collator=collate).train()

    lora = out / "lora"
    model.save_pretrained(str(lora))
    tok.save_pretrained(str(lora))
    gguf_dir = out / "gguf"
    try:
        model.save_pretrained_gguf(str(gguf_dir), tok, quantization_method="q4_k_m")
    except Exception as e:  # repli : fusion 16 bits, conversion par « omnifr export »
        print(f"export GGUF direct impossible ({e}) ; fusion 16 bits pour « omnifr export »", flush=True)
        model.save_pretrained_merged(str(out / "fusion16"), tok, save_method="merged_16bit")
    return out
