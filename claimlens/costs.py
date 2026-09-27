"""Cost per document, computed from what a run actually logged.

* Self-hosted: GPU $/hour x measured seconds per document (sequential, as in
  our runs -- a batched server would be several times cheaper per document).
* Hosted API: logged input/output tokens x $/million tokens.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional


def run_cost(run_dir: str | Path, gpu_hourly: Optional[float] = None, in_price: Optional[float] = None,
             out_price: Optional[float] = None) -> dict:
    recs = [json.loads(l) for l in open(Path(run_dir) / "audit.jsonl", encoding="utf-8") if l.strip()]
    lat = [r["latency_ms"] for r in recs if r.get("latency_ms") is not None]
    out = {"run": Path(run_dir).name, "docs": len(recs),
           "latency_s_p50": round(statistics.median(lat) / 1000, 1) if lat else None}
    if gpu_hourly is not None and lat:
        per_doc = gpu_hourly * statistics.mean(lat) / 1000 / 3600
        out.update({"gpu_hourly_usd": gpu_hourly, "usd_per_doc": round(per_doc, 5),
                    "usd_per_1000_docs": round(per_doc * 1000, 2)})
    toks_in = [r.get("input_tokens") for r in recs if r.get("input_tokens") is not None]
    toks_out = [r.get("output_tokens") for r in recs if r.get("output_tokens") is not None]
    if toks_in and toks_out:
        out.update({"input_tokens_mean": round(statistics.mean(toks_in)), "output_tokens_mean": round(statistics.mean(toks_out))})
        if in_price is not None and out_price is not None:
            per_doc = statistics.mean(toks_in) / 1e6 * in_price + statistics.mean(toks_out) / 1e6 * out_price
            out.update({"usd_per_doc": round(per_doc, 5), "usd_per_1000_docs": round(per_doc * 1000, 2),
                        "batch_api_usd_per_1000_docs": round(per_doc * 500, 2)})
    return out
