"""Phase 2: supervised fine-tuning (SFT) with LoRA.

The idea in one paragraph: show the model (image + prompt) and the exact answer
we want, and nudge its weights so that answer becomes more likely. We do not
change the 3 billion original weights. Instead LoRA adds small trainable
"adapter" matrices (~30M numbers) to the language part of the model; only those
are trained. The adapter is a separate ~120 MB file: easy to version, audit,
compare and roll back.

Design choices (each one maps to a Phase 1 finding):

* **Targets carry exact boxes around the VALUE**, never the label, which
  attacks the 59-65% "right value, wrong citation" rate.
* **Absent fields are explicitly null** in the targets, which is how the model
  learns that "no claim number on this invoice" is a valid answer (Phase 1:
  100% of hallucinations were the invoice number copied into claim_number).
* **Poor scans are oversampled** (default ~35% of examples vs ~20% in the raw
  data), because they caused ~half of all errors.
* **The prompt is the same v2 prompt used at evaluation time**, built by the
  same function, so training and inference see identical text.
* **Loss is computed only on the answer tokens**, not on the prompt/image.
* **The vision encoder is frozen**; LoRA goes on the language model only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Optional

from .prompts import PROMPT_VERSION, build_prompt
from .schema import FIELD_NAMES

# LoRA on the language model's attention + MLP projections, but NOT the vision
# encoder (whose MLP layers share the names gate_proj/up_proj/down_proj).
LORA_TARGET_REGEX = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


# --------------------------------------------------------------------------
# Training data
# --------------------------------------------------------------------------
def target_json(label: dict) -> str:
    """The exact answer we teach: v2 format, one field per line, integer boxes."""
    lines = []
    for name in FIELD_NAMES:
        f = label["fields"].get(name)
        if f is None:
            lines.append(f'  "{name}": null')
        else:
            obj = {"value": f["value"], "evidence_text": f["evidence_text"], "page": 1,
                   "bbox": [int(round(v)) for v in f["bbox"]]}
            lines.append(f'  "{name}": ' + json.dumps(obj, ensure_ascii=False, separators=(", ", ": ")))
    return "{\n" + ",\n".join(lines) + "\n}"


def build_examples(data_dir: str | Path, split: str = "train", poor_share: Optional[float] = 0.35,
                   seed: int = 0) -> list[dict]:
    """One example per document; poor-quality scans duplicated up to `poor_share`."""
    split_dir = Path(data_dir) / split
    examples = []
    with open(split_dir / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            label = json.loads((split_dir / row["label"]).read_text(encoding="utf-8"))
            examples.append({"doc_id": row["doc_id"], "image_path": str(split_dir / row["image"]),
                             "width": label["width"], "height": label["height"],
                             "target": target_json(label), "quality": row["quality"]})
    rng = random.Random(seed)
    if poor_share:
        poor = [e for e in examples if e["quality"] == "poor"]
        n_other = len(examples) - len(poor)
        want = math.ceil(poor_share * n_other / (1 - poor_share))  # poor / (poor + other) = share
        extra = [dict(e, doc_id=e["doc_id"] + f"#dup{i}") for i, e in
                 enumerate(rng.choices(poor, k=max(0, want - len(poor))))] if poor else []
        examples += extra
    rng.shuffle(examples)
    return examples


def describe_examples(examples: list[dict]) -> dict:
    n = len(examples)
    poor = sum(e["quality"] == "poor" for e in examples)
    nulls = sum(e["target"].count(": null") for e in examples)
    return {"n_examples": n, "unique_docs": len({e["doc_id"].split("#")[0] for e in examples}),
            "poor_share": round(poor / n, 3) if n else 0, "null_fields": nulls}


def encode_example(processor, example: dict, prompt_version: str = PROMPT_VERSION) -> dict:
    """Tokenise prompt and answer separately, then join them, so the boundary is
    exactly where generation starts at inference time. Labels are -100 (ignored)
    for every prompt/image token."""
    import torch
    from PIL import Image

    from .predictors import chat_prompt_text

    image = Image.open(example["image_path"]).convert("RGB")
    prompt = build_prompt(image.width, image.height, prompt_version)
    enc = processor(text=[chat_prompt_text(processor, prompt)], images=[image], return_tensors="pt")
    eos = "<|im_end|>"  # Qwen's end-of-turn token: we also teach the model to STOP
    tgt = processor.tokenizer(example["target"] + eos, add_special_tokens=False, return_tensors="pt")["input_ids"]
    enc["input_ids"] = torch.cat([enc["input_ids"], tgt], dim=1)
    enc["attention_mask"] = torch.ones_like(enc["input_ids"])
    enc["target_ids"] = tgt  # the tokens we compute loss on
    return dict(enc)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05, gradient_checkpointing: bool = True):
    import torch
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad_(False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGET_REGEX,
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for n, p in model.named_parameters():  # keep trainable weights in fp32 for stable fp16 training
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_pct": round(100 * trainable / total, 3)}


def answer_loss(model, batch: dict):
    """Cross-entropy on the answer tokens only. Asks the model for logits of the
    last T+1 positions (T = answer length) instead of all ~2,300 positions --
    the image alone is ~1,600 tokens -- which saves several GB of GPU memory."""
    import torch
    import torch.nn.functional as F

    target = batch["target_ids"]
    T = target.shape[1]
    inputs = {k: v for k, v in batch.items() if k != "target_ids"}
    try:
        logits = model(**inputs, logits_to_keep=T + 1, use_cache=False).logits
    except TypeError:  # older transformers
        logits = model(**inputs, use_cache=False).logits[:, -(T + 1):]
    logits = logits[:, :-1].float()  # position i predicts answer token i
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1).to(logits.device))


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
def train(model, processor, train_examples: list[dict], val_examples: Optional[list[dict]] = None, *,
          epochs: int = 2, lr: float = 1e-4, grad_accum: int = 8, warmup_frac: float = 0.05,
          max_grad_norm: float = 1.0, max_minutes: Optional[float] = None, log_every: int = 10,
          out_dir: Optional[str | Path] = None, seed: int = 0,
          on_log: Optional[Callable[[dict], None]] = None) -> list[dict]:
    """Plain PyTorch loop: batch size 1, gradient accumulation, fp16 mixed
    precision on GPU. Saves the adapter at the end of every epoch."""
    import torch

    torch.manual_seed(seed)
    device = next(p for p in model.parameters()).device
    use_amp = device.type == "cuda"
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    total_updates = max(1, math.ceil(len(train_examples) * epochs / grad_accum))
    warmup = max(1, int(warmup_frac * total_updates))

    def lr_lambda(step):  # linear warm-up, then cosine decay to 10%
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_updates - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, t0, update, running = [], time.time(), 0, []
    log = on_log or (lambda rec: print(json.dumps(rec)))
    model.train()

    stop = False
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_examples)))
        random.Random(seed + epoch).shuffle(order)
        for i, idx in enumerate(order, 1):
            batch = {k: v.to(device) for k, v in encode_example(processor, train_examples[idx]).items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = answer_loss(model, batch)
            if not math.isfinite(loss.item()):  # fp16 overflow guard: skip this example
                log({"note": f"non-finite loss on {train_examples[idx]['doc_id']} -- skipped"})
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss / grad_accum).backward()
            running.append(loss.item())
            if i % grad_accum == 0 or i == len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                update += 1
                if update % log_every == 0 or update == 1:
                    rec = {"epoch": epoch, "update": update, "of": total_updates,
                           "train_loss": round(sum(running) / len(running), 4),
                           "lr": round(sched.get_last_lr()[0], 7), "minutes": round((time.time() - t0) / 60, 1)}
                    history.append(rec)
                    log(rec)
                    running = []
            if max_minutes and (time.time() - t0) / 60 > max_minutes:
                log({"note": f"time budget of {max_minutes} min reached -- stopping early"})
                stop = True
                break
        rec = {"epoch": epoch, "end_of_epoch": True, "minutes": round((time.time() - t0) / 60, 1)}
        if val_examples:
            rec["val_loss"] = round(evaluate_loss(model, processor, val_examples), 4)
        history.append(rec)
        log(rec)
        if out_dir:
            save_adapter(model, out_dir, history=history)
        if stop:
            break
    model.eval()
    return history


def evaluate_loss(model, processor, examples: list[dict]) -> float:
    import torch

    device = next(p for p in model.parameters()).device
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for ex in examples:
            batch = {k: v.to(device) for k, v in encode_example(processor, ex).items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                losses.append(answer_loss(model, batch).item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


# --------------------------------------------------------------------------
# Saving, fingerprinting, training card
# --------------------------------------------------------------------------
def save_adapter(model, out_dir: str | Path, history: Optional[list] = None, card: Optional[dict] = None) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    if history is not None:
        (out / "training_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    if card is not None:
        card = dict(card, adapter_sha256=adapter_fingerprint(out))
        (out / "training_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return adapter_fingerprint(out)


def adapter_fingerprint(adapter_dir: str | Path) -> str:
    """Hash of the adapter weights -- goes into every audit record."""
    h = hashlib.sha256()
    for p in sorted(Path(adapter_dir).glob("adapter_model*")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]
