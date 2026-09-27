"""Build the web demo's replay bundle from real run outputs.

    python web/scripts/build_replay.py            (from the repo root, after the Kaggle runs are in runs/)

For a handful of chosen documents it copies the page image and writes one JSON
file holding, per model, exactly what the service would have returned:
fields with evidence, review flags, audit hashes, plus the evaluator's per-field
verdict against the ground truth. Nothing here is re-generated or edited: raw
outputs come straight from each run's hash-chained audit log.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from claimlens.evaluation import parse_prediction  # noqa: E402
from claimlens.service import review_flags  # noqa: E402

OUT = ROOT / "web" / "public" / "replay"

MODELS = {  # key -> (label, run prefix)
    "baseline": ("Baseline (prompt only)", "baseline_v2_greedy"),
    "sft_v2": ("SFT v2", "sft_v2ctrl_v2_greedy"),
    "grpo": ("GRPO", "grpo_v1_v2_greedy"),
}

# Chosen to show what the project found, good and bad (see docs/RESULTS.md).
DOCS = [
    ("test-00041", "Invoice with no claim number. The baseline copies the invoice number into it; SFT answers null. No review flag fires, because the copied value really is on the page."),
    ("test-00011", "Baseline invents a claim number that is not on the page; SFT does not."),
    ("test-00000", "Right values, wrong boxes: the baseline reads every field but its citations miss; after SFT every box lands."),
    ("test-00005", "Hard scan: a name and a long ID are misread by one character in every model (the vision encoder was frozen). The fine-tuned models' date slip is caught by the evidence-mismatch flag."),
    ("test-00089", "SFT v2 swaps two fields; GRPO gets this one right."),
    ("holdout-00000", "Unseen label \"Our reference\": the baseline and GRPO return no claim number; SFT v2, trained on randomised labels, finds it."),
    ("holdout-00035", "Unseen layout with a distractor ID; the baseline takes the bait, the fine-tuned models don't."),
    ("holdout-00068", "The baseline reads every value correctly but its boxes miss; GRPO abstains on the \"Our reference\" claim number."),
    ("holdout-00015", "Unseen invoice with no claim number: every model copies the account number into it, so the null behaviour learned in training did not transfer. It is flagged only because the date was cited from a line item."),
    ("holdout-00046", "Accented name: the diacritic is dropped even after fine-tuning."),
    ("holdout-00075", "SFT v2 swaps claim and policy numbers on an unseen layout; GRPO avoids it."),
    ("holdout-00036", "Degenerate output (a run of \"!\") from every model: unparseable, so the service routes it to a human."),
]


def _index(path: Path) -> dict[str, dict]:
    return {r["doc_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def main() -> None:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    runs, audits = {}, {}
    for key, (_, prefix) in MODELS.items():
        for split in ("test", "holdout"):
            d = ROOT / "runs" / f"{prefix}_{split}"
            runs[key, split] = _index(d / "results.jsonl")
            audits[key, split] = _index(d / "audit.jsonl")

    registry = json.loads((ROOT / "models" / "registry.json").read_text(encoding="utf-8"))
    docs = []
    for doc_id, note in DOCS:
        split = doc_id.split("-")[0]
        label = json.loads((ROOT / "data" / split / "labels" / f"{doc_id}.json").read_text(encoding="utf-8"))
        shutil.copy(ROOT / "data" / split / "images" / f"{doc_id}.jpg", OUT / "images" / f"{doc_id}.jpg")
        preds = {}
        for key, (name, prefix) in MODELS.items():
            a, r = audits[key, split][doc_id], runs[key, split][doc_id]
            parsed = parse_prediction(a["raw_output"])
            reasons = review_flags(parsed)
            preds[key] = {
                "response": {  # the same shape POST /v1/extract returns
                    "request_id": f"{a['run_id']}/{doc_id}",
                    "fields": {k: (v.model_dump() if v else None) for k, v in parsed["fields"].items()},
                    "review": {"needs_review": bool(reasons), "reasons": reasons},
                    "model": name,
                    "latency_ms": a["latency_ms"],
                    "audit": {"record_hash": a["record_hash"], "prev_hash": a["prev_hash"]},
                },
                "raw_output": a["raw_output"],
                "model_revision": a["model_revision"],
                "scores": {
                    "field_accuracy": r["field_accuracy"], "grounded_accuracy": r["grounded_accuracy"],
                    "all_correct": r["all_correct"],
                    "fields": {k: {"error_type": v["error_type"], "correct": v["correct"],
                                   "citation_ok": v.get("citation_ok"), "iou": v.get("iou")}
                               for k, v in r["fields"].items()},
                },
            }
        docs.append({
            "doc_id": doc_id, "split": split, "doc_type": label["doc_type"], "note": note,
            "image": f"replay/images/{doc_id}.jpg", "width": label["width"], "height": label["height"],
            "gold": label["fields"], "predictions": preds,
        })

    metrics = {k: {s: v["evaluations"].get(s) for s in ("test", "holdout", "fairness")}
               for k, v in registry["models"].items()}
    bundle = {"models": {k: v[0] for k, v in MODELS.items()}, "production": registry["production"],
              "metrics": metrics, "docs": docs}
    (OUT / "replay.json").write_text(json.dumps(bundle, indent=1), encoding="utf-8")
    print(f"wrote {len(docs)} documents to {OUT}")


if __name__ == "__main__":
    main()
