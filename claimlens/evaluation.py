"""Evaluation harness -- the most important code in the project.

The same scoring functions are used three ways:

1. **Evaluation** -- before/after numbers for the baseline, SFT and RL models.
2. **RL reward** -- ``compute_reward(raw_text, label)`` returns a number in
   [0, 1] that GRPO can maximise directly in Phase 3.
3. **Fairness** -- ``fairness_metrics`` compares results across synthetic name
   groups and across counterfactual sets (same document, different name).

Everything here is plain Python: it runs on a laptop CPU.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Optional

from pydantic import ValidationError

from .normalize import normalize_text, normalize_value, similarity
from .schema import FIELD_NAMES, ClaimExtraction, ExtractedField

# Bump whenever scoring rules change, and re-score ALL runs (baseline included)
# so before/after numbers stay comparable. History:
#   1.0  initial
#   1.1  IDs compared on letters+digits only ('POL. 214-4955 F' == 'POL-214-4955-F');
#        new error type 'distractor_for_absent' (field not on page, model copied
#        another value such as the invoice number) split out of 'hallucinated'.
EVALUATOR_VERSION = "1.1"

IOU_THRESHOLD = 0.5  # a citation "hits" if its box overlaps the true box by >= 50% IoU
NEAR_MISS = 0.8  # character similarity above which a wrong value counts as a near miss

# Reward weights (Phase 3 will optimise this number). Kept here, next to the
# metrics, so the reward and the evaluation can never silently diverge.
REWARD_WEIGHTS = {"format": 0.1, "accuracy": 0.6, "grounding": 0.3}


# --------------------------------------------------------------------------
# Parsing model output
# --------------------------------------------------------------------------
def extract_json_text(raw: str) -> Optional[str]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    # Accept a bare object, or an object wrapped in a list ("[{...}]").
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    if end <= start:
        return None
    return text[start:end + 1]


def parse_prediction(raw: str) -> dict[str, Any]:
    """Parse raw model text into per-field predictions.

    Returns a dict with:
      json_valid   -- the text contained a parseable JSON object
      schema_valid -- that object matched the schema exactly (strict)
      error        -- short reason when something failed
      fields       -- {field: ExtractedField | None}, parsed leniently so one
                      malformed field does not throw away the others
    """
    out: dict[str, Any] = {"json_valid": False, "schema_valid": False, "error": None,
                           "fields": {f: None for f in FIELD_NAMES}}
    js = extract_json_text(raw or "")
    if js is None:
        out["error"] = "no_json_found"
        return out
    try:
        obj = json.loads(js)
    except json.JSONDecodeError:
        out["error"] = "invalid_json"
        return out
    wrapped = False
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj, wrapped = obj[0], True  # readable, but not the format we asked for
    if not isinstance(obj, dict):
        out["error"] = "json_not_an_object"
        return out
    out["json_valid"] = True

    # {"value": null, ...} is a reasonable way to say "not present".
    cleaned = {k: (None if isinstance(v, dict) and v.get("value") in (None, "") else v) for k, v in obj.items()}
    try:
        ClaimExtraction.model_validate(cleaned)
        strict_ok = all(isinstance(cleaned.get(f), (dict, type(None))) for f in FIELD_NAMES)
        out["schema_valid"] = strict_ok and not wrapped
        if not out["schema_valid"]:
            out["error"] = "wrapped_in_list" if wrapped else "schema_error"
    except ValidationError:
        out["error"] = "schema_error"

    for f in FIELD_NAMES:
        v = cleaned.get(f)
        if v is None:
            continue
        if isinstance(v, dict):
            try:
                out["fields"][f] = ExtractedField.model_validate(v)
            except ValidationError:
                if v.get("value") is not None:  # keep the value, drop the broken evidence
                    out["fields"][f] = ExtractedField(value=str(v["value"]))
        elif isinstance(v, (str, int, float)):
            out["fields"][f] = ExtractedField(value=str(v))
    return out


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def iou(a: Optional[list[float]], b: Optional[list[float]]) -> float:
    if not a or not b:
        return 0.0
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _covered_fraction(seg: list[float], box: list[float]) -> float:
    ix0, iy0, ix1, iy1 = max(seg[0], box[0]), max(seg[1], box[1]), min(seg[2], box[2]), min(seg[3], box[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1e-6, (seg[2] - seg[0]) * (seg[3] - seg[1]))
    return inter / area


def text_in_box(layout: list[dict], box: Optional[list[float]], min_cover: float = 0.5) -> list[dict]:
    """Layout segments that lie (mostly) inside the cited box."""
    if not box:
        return []
    return [s for s in layout if _covered_fraction(s["bbox"], box) >= min_cover]


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# --------------------------------------------------------------------------
# Scoring one document
# --------------------------------------------------------------------------
def score_field(name: str, pred: Optional[ExtractedField], gold: Optional[dict], layout: list[dict]) -> dict:
    gold_val = gold["value"] if gold else None
    pred_val = pred.value if pred else None
    g_norm = normalize_value(name, gold_val)
    p_norm = normalize_value(name, pred_val)

    res = {
        "field": name,
        "gold_value": gold_val,
        "pred_value": pred_val,
        "gold_present": g_norm is not None,
        "pred_present": p_norm is not None,
        "correct": False,
        "error_type": None,
        "iou": 0.0,
        "citation_ok": False,
        "evidence_supported": False,
        "pred_bbox": pred.bbox if pred else None,
        "gold_bbox": gold["bbox"] if gold else None,
    }

    if g_norm is None and p_norm is None:
        res["correct"], res["error_type"] = True, "correct_null"
        return res
    if g_norm is None:
        # A value for a field that is not on the page. Did it copy another
        # value that IS on the page (e.g. the invoice number), or invent one?
        res["error_type"] = ("distractor_for_absent" if _matches_other_text(name, p_norm, layout)
                             else "hallucinated")
    elif p_norm is None:
        res["error_type"] = "missed"
    elif p_norm == g_norm:
        res["correct"], res["error_type"] = True, "correct"
    else:
        res["error_type"] = _classify_wrong_value(name, p_norm, g_norm, layout)

    # Evidence checks (only meaningful when the model gave a value).
    if pred is not None and p_norm is not None:
        if gold:
            res["iou"] = round(iou(pred.bbox, gold["bbox"]), 3)
            res["citation_ok"] = res["correct"] and res["iou"] >= IOU_THRESHOLD
        cited = text_in_box(layout, pred.bbox)
        cited_norm = [normalize_value(name, s["text"]) for s in cited]
        joined = normalize_text(" ".join(s["text"] for s in cited))
        res["evidence_supported"] = bool(cited) and (p_norm in cited_norm or (len(p_norm) > 3 and p_norm in joined))
    return res


def _matches_other_text(name: str, p_norm: str, layout: list[dict]) -> bool:
    for seg in layout:
        if seg["role"] in ("distractor", "value") and (
                normalize_value(name, seg["text"]) == p_norm or normalize_text(seg["text"]) == p_norm):
            return True
    return False


def _classify_wrong_value(name: str, p_norm: str, g_norm: str, layout: list[dict]) -> str:
    # Did the model read a *different* number/date that is also on the page
    # (subtotal instead of total, report date instead of loss date, ...)?
    for seg in layout:
        if seg["role"] == "distractor" and seg["field"] == name and normalize_value(name, seg["text"]) == p_norm:
            return "distractor_confusion"
    for seg in layout:
        if seg["role"] == "value" and seg["field"] not in (None, name):
            if normalize_value(seg["field"], seg["text"]) == p_norm or normalize_text(seg["text"]) == p_norm:
                return "field_swap"
    if _strip_accents(p_norm) == _strip_accents(g_norm):
        return "diacritics"  # e.g. 'Jose Garcia' for 'José García'
    if similarity(p_norm, g_norm) >= NEAR_MISS:
        return "near_miss"  # a character or two off -- typical OCR-style misread
    return "wrong_value"


def score_document(raw_text: str, label: dict) -> dict:
    """Score one model output against one ground-truth label."""
    parsed = parse_prediction(raw_text)
    layout = label.get("layout", [])
    fields = {f: score_field(f, parsed["fields"][f], label["fields"].get(f), layout) for f in FIELD_NAMES}
    if not parsed["json_valid"]:
        for r in fields.values():
            r["correct"], r["error_type"] = False, "unparseable"

    n = len(FIELD_NAMES)
    present = [r for r in fields.values() if r["gold_present"]]
    field_acc = sum(r["correct"] for r in fields.values()) / n
    grounded = [r["correct"] and r["citation_ok"] for r in present]
    grounding_score = (sum(
        (0.5 * r["evidence_supported"] + 0.5 * min(1.0, r["iou"] / IOU_THRESHOLD)) if r["correct"] else 0.0
        for r in present) / len(present)) if present else 1.0

    if parsed["json_valid"]:
        comps = {"format": 1.0 if parsed["schema_valid"] else 0.5, "accuracy": field_acc,
                 "grounding": grounding_score}
        reward = sum(REWARD_WEIGHTS[k] * v for k, v in comps.items())
    else:
        comps, reward = {"format": 0.0, "accuracy": 0.0, "grounding": 0.0}, 0.0

    return {
        "doc_id": label["doc_id"],
        "json_valid": parsed["json_valid"],
        "schema_valid": parsed["schema_valid"],
        "parse_error": parsed["error"],
        "field_accuracy": round(field_acc, 4),
        "grounded_accuracy": round(sum(grounded) / len(grounded), 4) if grounded else 1.0,
        "all_correct": all(r["correct"] for r in fields.values()),
        "reward": round(reward, 4),
        "reward_components": {k: round(v, 4) for k, v in comps.items()},
        "fields": fields,
    }


def compute_reward(raw_text: str, label: dict) -> float:
    """Single number in [0, 1] for RL (Phase 3). Higher is better."""
    return score_document(raw_text, label)["reward"]


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def field_rows(results: list[dict], manifest: dict[str, dict]) -> list[dict]:
    """Flatten to one row per (document, field) -- the shape error analysis wants."""
    rows = []
    for r in results:
        meta = manifest.get(r["doc_id"], {})
        for f in r["fields"].values():
            rows.append({
                "doc_id": r["doc_id"], "doc_type": meta.get("doc_type"),
                "line_of_business": meta.get("line_of_business"), "quality": meta.get("quality"),
                "handwritten": meta.get("handwritten"), "skew_deg": meta.get("skew_deg"),
                "group": meta.get("group"), "gender": meta.get("gender"), "set_id": meta.get("set_id"),
                "json_valid": r["json_valid"], **{k: v for k, v in f.items()},
            })
    return rows


def _rate(xs) -> Optional[float]:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def summary_metrics(results: list[dict], manifest: dict[str, dict]) -> dict:
    rows = field_rows(results, manifest)
    present = [r for r in rows if r["gold_present"]]
    absent = [r for r in rows if not r["gold_present"]]
    pred_given = [r for r in rows if r["pred_present"]]

    def slice_acc(key):
        groups = defaultdict(list)
        for r in rows:
            groups[str(r[key])].append(r["correct"])
        return {k: {"field_accuracy": _rate(v), "n_fields": len(v)} for k, v in sorted(groups.items())}

    per_field = {}
    for f in FIELD_NAMES:
        fr = [r for r in rows if r["field"] == f]
        fp = [r for r in fr if r["gold_present"]]
        per_field[f] = {"accuracy": _rate(r["correct"] for r in fr),
                        "grounded_accuracy": _rate(r["correct"] and r["citation_ok"] for r in fp),
                        "n_present": len(fp)}

    return {
        "n_docs": len(results),
        "json_valid_rate": _rate(r["json_valid"] for r in results),
        "schema_valid_rate": _rate(r["schema_valid"] for r in results),
        "field_accuracy": _rate(r["correct"] for r in rows),
        "grounded_accuracy": _rate(r["correct"] and r["citation_ok"] for r in present),
        "evidence_supported_rate": _rate(r["evidence_supported"] for r in pred_given),
        "hallucination_rate": _rate(r["pred_present"] for r in absent),
        "miss_rate": _rate(not r["pred_present"] for r in present),
        "doc_exact_match": _rate(r["all_correct"] for r in results),
        "mean_reward": _rate(r["reward"] for r in results),
        "per_field": per_field,
        "by_doc_type": slice_acc("doc_type"),
        "by_quality": slice_acc("quality"),
        "by_handwritten": slice_acc("handwritten"),
        "by_line_of_business": slice_acc("line_of_business"),
        "error_types": dict(Counter(r["error_type"] for r in rows if not r["correct"]).most_common()),
    }


def fairness_metrics(results: list[dict], manifest: dict[str, dict]) -> dict:
    """Two views of bias.

    * Group parity -- is accuracy similar across synthetic name groups/genders?
      (The claimant name itself is reported separately: reading some names
      worse than others is a fairness problem in its own right.)
    * Counterfactual consistency -- within a set of documents that differ ONLY
      in the claimant's name, do all the other extracted fields stay the same?
    """
    rows = field_rows(results, manifest)
    other = [r for r in rows if r["field"] != "claimant_name"]
    names = [r for r in rows if r["field"] == "claimant_name"]

    def parity(key):
        out = {}
        for g in sorted({r[key] for r in rows if r[key]}):
            out[g] = {"other_fields_accuracy": _rate(r["correct"] for r in other if r[key] == g),
                      "name_accuracy": _rate(r["correct"] for r in names if r[key] == g),
                      "n_docs": len({r["doc_id"] for r in rows if r[key] == g})}
        return out

    def gap(table, metric):
        vals = [v[metric] for v in table.values() if v[metric] is not None]
        return round(max(vals) - min(vals), 4) if len(vals) > 1 else None

    by_group, by_gender = parity("group"), parity("gender")

    # Counterfactual consistency
    sets = defaultdict(lambda: defaultdict(list))
    for r in other:
        if r["set_id"]:
            sets[r["set_id"]][r["field"]].append(normalize_value(r["field"], r["pred_value"]))
    flips = Counter()
    consistent_sets = 0
    for fields in sets.values():
        ok = True
        for f, vals in fields.items():
            if len(set(vals)) > 1:
                flips[f] += 1
                ok = False
        consistent_sets += ok
    n_sets = len(sets)

    return {
        "by_group": by_group,
        "by_gender": by_gender,
        "group_gap_other_fields": gap(by_group, "other_fields_accuracy"),
        "group_gap_name": gap(by_group, "name_accuracy"),
        "gender_gap_other_fields": gap(by_gender, "other_fields_accuracy"),
        "gender_gap_name": gap(by_gender, "name_accuracy"),
        "counterfactual_sets": n_sets,
        "counterfactual_consistency": round(consistent_sets / n_sets, 4) if n_sets else None,
        "counterfactual_flip_rate_by_field": {f: round(flips[f] / n_sets, 4) for f in FIELD_NAMES
                                              if f != "claimant_name"} if n_sets else {},
    }
