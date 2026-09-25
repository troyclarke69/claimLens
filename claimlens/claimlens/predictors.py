"""Predictors: anything that turns a document image into raw model text.

* ``HFVisionPredictor`` -- a real open vision-language model from Hugging Face
  (default: Qwen2.5-VL-3B-Instruct). Needs a GPU (e.g. a free Kaggle T4).
* ``SimulatedPredictor`` -- NOT a model. It copies the ground truth and injects
  known, realistic errors at known rates. It exists so you can exercise the
  whole pipeline (evaluation, audit log, fairness report, error-analysis
  notebook) on a laptop CPU in seconds, and check that the harness catches the
  errors we planted. Never report its numbers as model results.

All predictors return raw text; parsing and scoring happen in evaluation.py,
exactly as they will for real models.
"""

from __future__ import annotations

import json
import random
import time
import unicodedata
from pathlib import Path
from typing import Optional

from .generator import _stable_seed
from .normalize import normalize_value
from .prompts import PROMPT_VERSION, build_prompt
from .schema import FIELD_NAMES


class Predictor:
    model_id: str = "base"
    model_revision: str = "n/a"
    prompt_version: str = PROMPT_VERSION

    def predict(self, image_path: Path, label: dict) -> dict:
        """Return {"raw_text", "latency_ms", ...}. Real models may only use the
        label for the page width/height -- never for the answer."""
        raise NotImplementedError


# --------------------------------------------------------------------------
class SimulatedPredictor(Predictor):
    """Ground truth + planted errors. For pipeline testing only."""

    def __init__(self, seed: int = 7, bias_group: Optional[str] = None, bias_strength: float = 0.0):
        self.seed = seed
        self.bias_group = bias_group
        self.bias_strength = bias_strength
        self.model_id = "simulated-predictor"
        self.model_revision = f"seed{seed}-bias_{bias_group}_{bias_strength}"

    def predict(self, image_path: Path, label: dict) -> dict:
        t0 = time.perf_counter()
        rng = random.Random(_stable_seed(self.seed, label["doc_id"]))
        meta = label["meta"]
        # Fairness variants share one underlying document, so the "reading"
        # randomness for non-name fields is keyed on the set, not the variant.
        # Any difference between variants therefore comes from planted bias.
        base_key = meta.get("set_id") or label["doc_id"]
        layout = label["layout"]
        p_err = {"clean": 0.03, "moderate": 0.08, "poor": 0.18}[meta["quality"]]
        if meta["handwritten"]:
            p_err += 0.12

        out = {}
        for f in FIELD_NAMES:
            frng = random.Random(_stable_seed(self.seed, base_key, f))
            if f == "claimant_name":
                frng = random.Random(_stable_seed(self.seed, label["doc_id"], f))
            out[f] = self._field(f, label["fields"].get(f), label, layout, frng, p_err)

        if self.bias_group and meta.get("group") == self.bias_group:
            # Planted bias: for the chosen group, the "model" also gets the
            # total wrong more often. The fairness report should flag this.
            if rng.random() < self.bias_strength and out.get("total_amount"):
                v = out["total_amount"]["value"]
                out["total_amount"]["value"] = _typo(v, rng, digits=True)

        text = json.dumps(out, ensure_ascii=False, indent=1)
        if random.Random(_stable_seed(self.seed, base_key, "truncate")).random() < 0.03:
            text = text[: int(len(text) * 0.6)]  # truncated output -> invalid JSON
        return {"raw_text": text, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    def _field(self, f, gold, label, layout, rng, p_err):
        doc_type = label["doc_type"]
        if gold is None:
            if rng.random() < 0.05:  # hallucinate a value that is not on the page
                fake = {"incident_type": "Auto collision", "policy_number": "POL-000-0000-A",
                        "claim_number": "CLM-2025-000001", "provider_name": "Unknown Provider"}.get(f, "N/A")
                return {"value": fake, "evidence_text": fake, "page": 1, "bbox": [60, 60, 200, 80]}
            return None

        pred = {"value": gold["value"], "evidence_text": gold["evidence_text"], "page": 1,
                "bbox": list(gold["bbox"])}

        # Distractor confusions -- the classic document-AI mistakes.
        if f == "total_amount" and doc_type == "invoice" and rng.random() < 0.15:
            seg = _find(layout, lambda s: s["role"] == "distractor" and s["field"] == f, last=True)
            if seg:
                return _from_seg(f, seg)
        if f == "date_of_loss" and rng.random() < 0.08:
            seg = _find(layout, lambda s: s["role"] == "distractor" and s["field"] == f)
            if seg:
                return _from_seg(f, seg)

        if f == "claimant_name" and _has_accents(pred["value"]) and rng.random() < 0.35:
            pred["value"] = _strip_accents(pred["value"])  # a very common real-world failure

        r = rng.random()
        if r < p_err * 0.25:
            return None  # missed
        if r < p_err:
            pred["value"] = _typo(pred["value"], rng, digits=f in ("total_amount", "claim_number", "policy_number"))
        if rng.random() < 0.06:  # right value, wrong place
            pred["bbox"] = [pred["bbox"][0], pred["bbox"][1] + 45, pred["bbox"][2], pred["bbox"][3] + 45]
        return pred


def _find(layout, cond, last=False):
    hits = [s for s in layout if cond(s)]
    return (hits[-1] if last else hits[0]) if hits else None


def _from_seg(f, seg):
    val = normalize_value(f, seg["text"]) if f in ("total_amount", "date_of_loss") else seg["text"]
    return {"value": val, "evidence_text": seg["text"], "page": 1, "bbox": list(seg["bbox"])}


def _has_accents(s):
    return any(unicodedata.combining(c) for c in unicodedata.normalize("NFKD", s))


def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _typo(s: str, rng: random.Random, digits=False) -> str:
    if not s:
        return s
    idx = [i for i, c in enumerate(s) if c.isalnum()]
    if not idx:
        return s
    i = rng.choice(idx)
    if s[i].isdigit():
        swap = {"0": "8", "1": "7", "3": "8", "5": "6", "6": "5", "7": "1", "8": "3", "9": "4", "2": "7", "4": "9"}
        c = swap[s[i]]
    else:
        c = rng.choice("aeilnorstu") if s[i].islower() else rng.choice("AEILNORSTU")
    return s[:i] + c + s[i + 1:]


# --------------------------------------------------------------------------
class HFVisionPredictor(Predictor):
    """A Hugging Face vision-language model (Qwen2.5-VL, SmolVLM, ...).

    Imports torch/transformers lazily so the rest of the project runs without
    them. Tested API pattern: transformers >= 4.49 (AutoModelForImageTextToText).
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", max_new_tokens: int = 900,
                 dtype: str = "float16", load_in_4bit: bool = False, max_pixels: Optional[int] = None,
                 adapter_path: Optional[str] = None, prompt_version: str = PROMPT_VERSION,
                 finish_json: bool = False):
        """finish_json=True blocks the end-of-turn token until every { and [ the
        model opened has been closed (a light form of constrained decoding)."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.prompt_version = prompt_version
        self.finish_json = finish_json
        self.decoding = "greedy+finish_json" if finish_json else "greedy"
        proc_kwargs = {}
        if max_pixels:
            proc_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)

        kwargs = {"device_map": "auto" if torch.cuda.is_available() else None}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = getattr(torch, dtype) if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        if adapter_path:  # Phase 2: LoRA adapter produced by SFT
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        cfg = getattr(self.model, "config", None)
        self.model_revision = getattr(cfg, "_commit_hash", None) or "unknown"
        if adapter_path:
            self.model_revision += f"+adapter:{Path(adapter_path).name}"

    @classmethod
    def from_model(cls, model, processor, model_id: str, model_revision: str, max_new_tokens: int = 900,
                   prompt_version: str = PROMPT_VERSION) -> "HFVisionPredictor":
        """Wrap a model that is already in memory (e.g. right after SFT training)."""
        import torch

        self = cls.__new__(cls)
        self.torch, self.model, self.processor = torch, model, processor
        self.model_id, self.model_revision = model_id, model_revision
        self.max_new_tokens, self.prompt_version = max_new_tokens, prompt_version
        self.finish_json, self.decoding = False, "greedy"
        return self

    def predict(self, image_path: Path, label: dict) -> dict:
        from PIL import Image

        torch = self.torch
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        prompt = build_prompt(w, h, self.prompt_version)
        text = chat_prompt_text(self.processor, prompt)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            extra = {}
            if self.finish_json:
                from transformers import LogitsProcessorList
                eos = self.model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                extra["logits_processor"] = LogitsProcessorList([_FinishJson(
                    self.processor.tokenizer, inputs["input_ids"].shape[1], eos)])
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, **extra)
        latency = (time.perf_counter() - t0) * 1000
        gen = out[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(gen, skip_special_tokens=True)[0]

        # Qwen2.5-VL answers in the pixel coordinates of the (possibly resized)
        # image it actually saw. Map boxes back to original-image pixels.
        scale, original = None, None
        if "image_grid_thw" in inputs:
            _, gh, gw = inputs["image_grid_thw"][0].tolist()
            rw, rh = gw * 14, gh * 14
            if (rw, rh) != (w, h):
                scale = (w / rw, h / rh)
                original, raw = raw, _rescale_bboxes(raw, *scale)
        return {"raw_text": raw, "latency_ms": round(latency, 1), "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": int(gen.shape[1]), "bbox_scale": scale,
                "raw_text_before_rescale": original, "decoding": self.decoding}


def chat_prompt_text(processor, prompt: str) -> str:
    """The exact text the model sees before it starts answering (shared by
    inference and SFT training, so the two can never drift apart)."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


class _FinishJson:
    """Logits processor: forbid stopping while JSON brackets are still open."""

    def __init__(self, tokenizer, prompt_len: int, eos_ids: list[int]):
        self.tok, self.prompt_len, self.eos = tokenizer, prompt_len, eos_ids

    def __call__(self, input_ids, scores):
        text = self.tok.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        depth = text.count("{") + text.count("[") - text.count("}") - text.count("]")
        if "{" not in text or depth > 0:
            scores[:, self.eos] = float("-inf")
        return scores


def _rescale_bboxes(raw: str, sx: float, sy: float) -> str:
    from .evaluation import extract_json_text

    js = extract_json_text(raw)
    if not js:
        return raw
    try:
        obj = json.loads(js)
    except json.JSONDecodeError:
        return raw
    for v in obj.values():
        if isinstance(v, dict) and isinstance(v.get("bbox"), list) and len(v["bbox"]) == 4:
            try:
                x0, y0, x1, y1 = (float(t) for t in v["bbox"])
                v["bbox"] = [round(x0 * sx, 1), round(y0 * sy, 1), round(x1 * sx, 1), round(y1 * sy, 1)]
            except (TypeError, ValueError):
                pass
    return json.dumps(obj, ensure_ascii=False)
