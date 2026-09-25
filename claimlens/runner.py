"""Run a predictor over a dataset split, log everything, then score it.

Output of one run (``runs/<run_id>/``):
  config.json          what was run: model, prompt, dataset hash, code version
  audit.jsonl          one tamper-evident record per document (the source of truth)
  results.jsonl        per-document scores
  scored_fields.jsonl  one row per (document, field) -- input to error analysis
  metrics.json         summary + fairness metrics
  report.md            human-readable report
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import audit
from .evaluation import EVALUATOR_VERSION, fairness_metrics, field_rows, score_document, summary_metrics
from .predictors import Predictor
from .prompts import prompt_hash
from .report import write_report


def load_manifest(split_dir: Path) -> dict[str, dict]:
    rows = {}
    with open(split_dir / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["doc_id"]] = r
    return rows


def load_label(split_dir: Path, doc_id: str) -> dict:
    return json.loads((split_dir / "labels" / f"{doc_id}.json").read_text(encoding="utf-8"))


def _dataset_hash(data_dir: Path, split: str) -> str:
    info_path = data_dir / "dataset_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text())["split_hashes"].get(split, "unknown")
    return "unknown"


def make_run_id(model_id: str, split: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short = re.sub(r"[^A-Za-z0-9.-]+", "-", model_id.split("/")[-1])[:40]
    return f"{stamp}_{short}_{split}"


def run_predictions(predictor: Predictor, data_dir: str | Path, split: str, runs_dir: str | Path = "runs",
                    run_id: Optional[str] = None, limit: Optional[int] = None, verbose: bool = True) -> Path:
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    manifest = load_manifest(split_dir)
    doc_ids = list(manifest)[:limit] if limit else list(manifest)

    run_id = run_id or make_run_id(predictor.model_id, split)
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ds_hash = _dataset_hash(data_dir, split)
    config = {
        "run_id": run_id, "model_id": predictor.model_id, "model_revision": predictor.model_revision,
        "prompt_version": predictor.prompt_version, "prompt_hash": prompt_hash(version=predictor.prompt_version),
        "decoding": getattr(predictor, "decoding", "n/a"), "data_dir": str(data_dir),
        "split": split, "dataset_hash": ds_hash, "n_docs": len(doc_ids), "code_version": audit.code_version(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    log = audit.AuditLog(run_dir / "audit.jsonl")
    done = set()
    if log.path.exists():  # resume support: Kaggle sessions can stop mid-run
        with open(log.path, encoding="utf-8") as f:
            done = {json.loads(line)["doc_id"] for line in f if line.strip()}

    for i, doc_id in enumerate(doc_ids, 1):
        if doc_id in done:
            continue
        label = load_label(split_dir, doc_id)
        img_path = split_dir / label["image"]
        pred = predictor.predict(img_path, label)
        log.append({
            "run_id": run_id, "doc_id": doc_id, "split": split, "image_sha256": audit.sha256_file(img_path),
            "dataset_hash": ds_hash, "model_id": predictor.model_id, "model_revision": predictor.model_revision,
            "prompt_version": predictor.prompt_version, "prompt_hash": config["prompt_hash"],
            "code_version": config["code_version"], "raw_output": pred.pop("raw_text"), **pred,
        })
        if verbose and (i % 10 == 0 or i == len(doc_ids)):
            print(f"  [{i}/{len(doc_ids)}] {doc_id}  ({pred.get('latency_ms')} ms)", flush=True)
    return run_dir


def evaluate_run(run_dir: str | Path, data_dir: Optional[str | Path] = None) -> dict:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    data_dir = Path(data_dir or config["data_dir"])
    split_dir = data_dir / config["split"]
    manifest = load_manifest(split_dir)

    ok, msg = audit.verify(run_dir / "audit.jsonl")
    results = []
    with open(run_dir / "audit.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            label = load_label(split_dir, rec["doc_id"])
            res = score_document(rec["raw_output"], label)
            res["latency_ms"] = rec.get("latency_ms")
            results.append(res)

    manifest_meta = {k: v for k, v in manifest.items()}
    # Keep the previous scoring if the rules changed, so nothing is silently overwritten.
    old_path = run_dir / "metrics.json"
    if old_path.exists():
        old_ver = json.loads(old_path.read_text(encoding="utf-8")).get("evaluator_version", "1.0")
        if old_ver != EVALUATOR_VERSION:
            for name in ("metrics.json", "report.md"):
                src = run_dir / name
                if src.exists():
                    src.replace(run_dir / f"{src.stem}_eval-{old_ver}{src.suffix}")

    metrics = {
        "run_id": config["run_id"], "model_id": config["model_id"], "split": config["split"],
        "evaluator_version": EVALUATOR_VERSION,
        "audit_chain_ok": ok, "audit_message": msg,
        "summary": summary_metrics(results, manifest_meta),
        "fairness": fairness_metrics(results, manifest_meta),
    }
    lat = sorted(r["latency_ms"] for r in results if r.get("latency_ms") is not None)
    if lat:
        metrics["summary"]["latency_ms_p50"] = lat[len(lat) // 2]
        metrics["summary"]["latency_ms_p95"] = lat[min(len(lat) - 1, int(len(lat) * 0.95))]

    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(run_dir / "scored_fields.jsonl", "w", encoding="utf-8") as f:
        for row in field_rows(results, manifest_meta):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(run_dir / "report.md", config, metrics)
    return metrics
