"""Phase 3: reinforcement learning with GRPO (Group Relative Policy Optimisation).

The idea in one paragraph: for each training document the model writes several
answers (a "group"), with a little randomness so they differ. Each answer is
scored with the cost-weighted reward from evaluation.py. Answers that beat the
group average are made more likely; answers below it, less likely. No answer
key is imitated -- the model is pushed toward whatever *scores* better, which
is how we can encode "a wrong value is worse than an honest null".

A second term (KL penalty) keeps the model close to where it started (the SFT
model), so it cannot drift into strange outputs that happen to game the reward.

Everything that could reveal reward hacking is logged per step: rewards, how
many fields came back null, box sizes, answer length and KL.

Setup used in this project: the SFT adapter is merged into the base model
first; GRPO then trains a *fresh* LoRA on top. The reference model for the KL
penalty is simply the same network with that fresh LoRA switched off.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Callable, Optional

from .evaluation import costed_reward
from .prompts import PROMPT_VERSION, build_prompt


def encode_prompt(processor, example: dict, prompt_version: str = PROMPT_VERSION) -> dict:
    from PIL import Image

    from .predictors import chat_prompt_text

    image = Image.open(example["image_path"]).convert("RGB")
    prompt = build_prompt(image.width, image.height, prompt_version)
    return dict(processor(text=[chat_prompt_text(processor, prompt)], images=[image], return_tensors="pt"))


def _repeat(enc: dict, n: int) -> dict:
    import torch

    out = {}
    for k, v in enc.items():
        if not torch.is_tensor(v):
            out[k] = v
        elif k == "pixel_values":
            out[k] = torch.cat([v] * n, dim=0)
        elif k == "image_grid_thw":
            out[k] = v.repeat(n, 1)
        elif v.dim() == 2 and v.shape[0] == 1:
            out[k] = v.repeat(n, 1)
        else:
            out[k] = v
    return out


def _with_completion(enc: dict, completion) -> dict:
    """Prompt + one completion, in the same shape sft.encode_example produces."""
    import torch

    n = enc["input_ids"].shape[1]
    tgt = completion.view(1, -1)
    batch = {}
    for k, v in enc.items():
        if k in ("pixel_values", "image_grid_thw") or not torch.is_tensor(v):
            batch[k] = v
        elif k == "input_ids":
            batch[k] = torch.cat([v, tgt.to(v.device)], dim=1)
        elif v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == n:
            batch[k] = torch.cat([v, torch.zeros((1, tgt.shape[1]), dtype=v.dtype, device=v.device)], dim=1)
        else:
            batch[k] = v
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    batch["target_ids"] = tgt
    return batch


def token_logprobs(model, batch: dict):
    """Log-probability of each completion token (shape [T])."""
    import torch

    target = batch["target_ids"]
    T = target.shape[1]
    inputs = {k: v for k, v in batch.items() if k != "target_ids"}
    try:
        logits = model(**inputs, logits_to_keep=T + 1, use_cache=False).logits
    except TypeError:
        logits = model(**inputs, use_cache=False).logits[:, -(T + 1):]
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return logp.gather(-1, target.to(logp.device).unsqueeze(-1)).squeeze(-1).squeeze(0)


def _cut_at_eos(ids, eos_ids: list[int]):
    for i, t in enumerate(ids.tolist()):
        if t in eos_ids:
            return ids[: i + 1]
    return ids


def _sample_stats(doc_res: dict) -> dict:
    """Signals that expose reward hacking: null answers and box sizes."""
    fields = doc_res["fields"].values()
    nulls = sum(not f["pred_present"] for f in fields)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for f in fields if (b := f.get("pred_bbox"))]
    return {"null_fields": nulls, "mean_box_area": (sum(areas) / len(areas)) if areas else 0.0}


def grpo_train(model, processor, examples: list[dict], labels: dict[str, dict], *,
               group_size: int = 4, lr: float = 2e-5, beta: float = 0.04, temperature: float = 0.9,
               top_p: float = 1.0, max_new_tokens: int = 600, prompts_per_update: int = 2, min_std: float = 0.1,
               costs: Optional[dict] = None, max_minutes: Optional[float] = None,
               out_dir: Optional[str | Path] = None, save_every: int = 20, seed: int = 0,
               reward_fn: Optional[Callable[[str, dict], dict]] = None,
               on_log: Optional[Callable[[dict], None]] = None) -> list[dict]:
    """Run GRPO over `examples` (one pass). `labels[doc_id]` is the ground truth,
    used ONLY inside the reward -- the model never sees it."""
    import torch

    from .sft import save_adapter

    torch.manual_seed(seed)
    random.seed(seed)
    reward_fn = reward_fn or (lambda text, label: costed_reward(text, label, costs))
    log = on_log or (lambda rec: print(json.dumps(rec)))
    device = next(p for p in model.parameters()).device
    use_amp = device.type == "cuda"
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen_cfg = model.generation_config if hasattr(model, "generation_config") else model.base_model.generation_config
    eos = gen_cfg.eos_token_id if isinstance(gen_cfg.eos_token_id, list) else [gen_cfg.eos_token_id]
    pad = getattr(processor.tokenizer, "pad_token_id", None) or eos[0]
    history, t0, pending = [], time.time(), 0

    for step, ex in enumerate(examples, 1):
        label = labels[ex["doc_id"].split("#")[0]]
        enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in encode_prompt(processor, ex).items()}
        L = enc["input_ids"].shape[1]

        # 1) sample a group of answers (no gradients)
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model.generate(**_repeat(enc, group_size), do_sample=True, temperature=temperature, top_p=top_p,
                                 max_new_tokens=max_new_tokens, pad_token_id=pad)
        completions = [_cut_at_eos(row[L:], eos) for row in out]
        texts = [processor.tokenizer.decode(c, skip_special_tokens=True) for c in completions]

        # 2) score them with the reward
        scored = [reward_fn(t, label) for t in texts]
        rewards = [s["reward"] for s in scored]
        mean_r = statistics.mean(rewards)
        std_r = statistics.pstdev(rewards)
        stats = [_sample_stats(s["doc"]) for s in scored]
        rec = {"step": step, "doc_id": ex["doc_id"], "reward_mean": round(mean_r, 4), "reward_std": round(std_r, 4),
               "rewards": [round(r, 3) for r in rewards],
               "null_fields_mean": round(statistics.mean(s["null_fields"] for s in stats), 2),
               "box_area_mean": round(statistics.mean(s["mean_box_area"] for s in stats), 0),
               "answer_tokens_mean": round(statistics.mean(len(c) for c in completions), 0)}

        # 3) policy-gradient update on the group (skip if all answers scored the same)
        if std_r < 1e-6:
            rec["note"] = "no learning signal (all answers scored the same)"
        else:
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            kls = []
            # Advantage = how much better than the group average. Dividing by the
            # group's spread is standard GRPO, but when all answers are nearly
            # identical (e.g. rewards 0.994 vs 0.938) it would blow tiny, noisy
            # differences up into full-strength updates. The floor (`min_std`)
            # keeps small differences small; real mistakes still get a strong signal.
            scale = max(std_r, min_std)
            for comp, r in zip(completions, rewards):
                adv = (r - mean_r) / scale
                batch = _with_completion(enc, comp)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    with torch.no_grad(), model.disable_adapter():
                        ref_lp = token_logprobs(model, batch)
                    pol_lp = token_logprobs(model, batch)
                    delta = ref_lp - pol_lp
                    kl = (torch.exp(delta) - delta - 1).mean()  # k3 estimator, always >= 0
                    loss = -(adv * pol_lp.mean()) + beta * kl
                if not math.isfinite(loss.item()):
                    continue
                scaler.scale(loss / (group_size * prompts_per_update)).backward()
                kls.append(kl.item())
            rec["kl"] = round(statistics.mean(kls), 5) if kls else None
            pending += 1
            if pending >= prompts_per_update:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                pending = 0
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        history.append(rec)
        log(rec)
        if out_dir and step % save_every == 0:
            save_adapter(model, out_dir, history=history)
        if max_minutes and rec["minutes"] > max_minutes:
            log({"note": f"time budget of {max_minutes} min reached -- stopping after {step} prompts"})
            break

    if pending:  # flush a half-filled update
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    model.eval()
    return history


def summarize(history: list[dict], window: int = 10) -> dict:
    """First vs last `window` steps: did reward rise, and did anything suspicious rise with it?"""
    steps = [h for h in history if "reward_mean" in h]
    if not steps:
        return {}
    first, last = steps[:window], steps[-window:]
    avg = lambda rows, k: round(statistics.mean(r[k] for r in rows if r.get(k) is not None), 4) if rows else None
    return {k: {"first": avg(first, k), "last": avg(last, k)}
            for k in ("reward_mean", "reward_std", "null_fields_mean", "box_area_mean", "answer_tokens_mean", "kl")}
