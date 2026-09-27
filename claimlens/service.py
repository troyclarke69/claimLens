"""Phase 4: a small HTTP service around the extractor.

    uvicorn claimlens.service:app --port 8000        (or: docker run ...)

Endpoints
  POST /v1/extract       upload a document image -> fields + evidence + review flags + audit receipt
  GET  /v1/model         which model is serving (name, base revision, adapter fingerprints)
  GET  /v1/audit/verify  re-check the audit log's hash chain
  GET  /v1/health

Configuration (environment variables)
  CLAIMLENS_PREDICTOR   simulated | hf | anthropic        (default: simulated -- laptop demo)
  CLAIMLENS_MODEL       registry model name to serve for hf (default: the registry's production model)
  CLAIMLENS_ADAPTERS    folder holding adapters/<name>     (default: adapters)
  CLAIMLENS_DATA        dataset folder (simulated mode looks up labels by doc_id)
  CLAIMLENS_AUDIT_LOG   path of the service audit log     (default: runs/service/audit.jsonl)

Design notes
* The model's answer is never trusted blindly: every response carries
  `review.needs_review` and the reasons (unparseable output, a required field
  missing, a value without a citation, a value that does not match its own
  evidence text). In a claims workflow those go to a human queue.
* Every request is written to the same hash-chained audit log used in
  evaluation; the response returns the record hash as a receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import __version__, audit
from .evaluation import EVALUATOR_VERSION, parse_prediction
from .normalize import normalize_value
from .prompts import PROMPT_VERSION, prompt_hash

REQUIRED_FIELDS = ("claimant_name", "date_of_loss", "total_amount")


def review_flags(parsed: dict) -> list[str]:
    """Reasons a human should look at this extraction before it is used."""
    reasons = []
    if not parsed["json_valid"]:
        return ["model output could not be parsed"]
    if not parsed["schema_valid"]:
        reasons.append("output did not match the schema exactly")
    for name, f in parsed["fields"].items():
        if f is None:
            if name in REQUIRED_FIELDS:
                reasons.append(f"{name}: required field not found")
            continue
        if not f.bbox:
            reasons.append(f"{name}: value has no citation box")
        if f.evidence_text and normalize_value(name, f.evidence_text) != normalize_value(name, f.value):
            reasons.append(f"{name}: value does not match its own evidence text")
    return reasons


class _State:
    predictor = None
    model_info: dict = {}
    log: Optional[audit.AuditLog] = None
    data_dir: Optional[Path] = None


state = _State()


def _load_predictor():
    kind = os.environ.get("CLAIMLENS_PREDICTOR", "simulated")
    if kind == "simulated":
        from .predictors import SimulatedPredictor
        state.data_dir = Path(os.environ.get("CLAIMLENS_DATA", "data"))
        return SimulatedPredictor(), {"name": "simulated-predictor", "note": "demo only: needs doc_id of a dataset document"}
    if kind == "anthropic":
        from .predictors import AnthropicPredictor
        p = AnthropicPredictor(os.environ.get("CLAIMLENS_API_MODEL", "claude-haiku-4-5"))
        return p, {"name": p.model_id, "hosted": True}
    if kind == "hf":
        from .predictors import HFVisionPredictor
        from .registry import REGISTRY_PATH
        reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        name = os.environ.get("CLAIMLENS_MODEL") or reg["production"]
        entry = reg["models"][name]
        root = Path(os.environ.get("CLAIMLENS_ADAPTERS", "adapters"))
        paths = [str(root / a["name"]) for a in entry["adapters"]]
        p = HFVisionPredictor(entry["base_model"], adapter_path=paths or None,
                              expected_fingerprints=[a["sha256"] for a in entry["adapters"]])
        return p, {"name": name, "base_model": entry["base_model"], "adapters": entry["adapters"]}
    raise RuntimeError(f"unknown CLAIMLENS_PREDICTOR={kind}")


@asynccontextmanager
async def _lifespan(_app):
    state.predictor, state.model_info = _load_predictor()   # load the model once, at startup
    state.log = audit.AuditLog(os.environ.get("CLAIMLENS_AUDIT_LOG", "runs/service/audit.jsonl"))
    yield


app = FastAPI(title="ClaimLens", version=__version__, lifespan=_lifespan,
              description="Evidence-cited extraction from insurance claim documents (demo).")


@app.get("/v1/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/v1/model")
def model():
    return {**state.model_info, "model_revision": getattr(state.predictor, "model_revision", None),
            "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash(), "evaluator_version": EVALUATOR_VERSION,
            "code_version": __version__}


@app.get("/v1/audit/verify")
def verify_audit():
    ok, msg = audit.verify(state.log.path) if state.log.path.exists() else (True, "empty log")
    return {"ok": ok, "message": msg}


@app.post("/v1/extract")
async def extract(file: UploadFile = File(...), doc_id: Optional[str] = Form(None)):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(400, "empty file")
    request_id = str(uuid.uuid4())
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / f"doc{suffix}"
        img_path.write_bytes(raw_bytes)
        label = _label_for(doc_id, img_path)
        t0 = time.perf_counter()
        out = state.predictor.predict(img_path, label)
        latency = round((time.perf_counter() - t0) * 1000, 1)

    parsed = parse_prediction(out["raw_text"])
    reasons = review_flags(parsed)
    fields = {k: (v.model_dump() if v else None) for k, v in parsed["fields"].items()}
    rec = state.log.append({
        "request_id": request_id, "image_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "filename": file.filename, "model": state.model_info.get("name"),
        "model_revision": getattr(state.predictor, "model_revision", None), "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(), "code_version": __version__, "raw_output": out["raw_text"],
        "latency_ms": latency, "needs_review": bool(reasons), "review_reasons": reasons,
    })
    return {"request_id": request_id, "fields": fields,
            "review": {"needs_review": bool(reasons), "reasons": reasons},
            "model": state.model_info.get("name"), "latency_ms": latency,
            "audit": {"record_hash": rec["record_hash"], "prev_hash": rec["prev_hash"]}}


def _label_for(doc_id: Optional[str], img_path: Path) -> dict:
    """Real models only need the page size. The simulated demo needs the label."""
    if state.data_dir is not None:
        if not doc_id:
            raise HTTPException(400, "simulated mode: pass doc_id of a dataset document (e.g. test-00000)")
        for split_dir in state.data_dir.iterdir():
            p = split_dir / "labels" / f"{doc_id}.json"
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        raise HTTPException(404, f"doc_id {doc_id} not found under {state.data_dir}")
    from PIL import Image
    w, h = Image.open(img_path).size
    return {"doc_id": doc_id or "upload", "width": w, "height": h}
