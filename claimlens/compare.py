"""Side-by-side comparison of runs (e.g. baseline vs SFT) from their metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

HEADLINE = [
    ("json_valid_rate", "Valid JSON", True),
    ("field_accuracy", "Field accuracy", True),
    ("grounded_accuracy", "Grounded accuracy", True),
    ("evidence_supported_rate", "Evidence supported", True),
    ("hallucination_rate", "Hallucination rate", False),
    ("miss_rate", "Miss rate", False),
    ("doc_exact_match", "Document exact match", True),
    ("mean_reward", "Mean reward", True),
]


def _load(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    m["_dir"] = run_dir.name
    return m


def compare(run_dirs: list[str | Path]) -> str:
    runs = [_load(r) for r in run_dirs]
    versions = {r.get("evaluator_version", "1.0") for r in runs}
    L = []
    if len(versions) > 1:
        L.append(f"> ⚠️ Runs were scored with different evaluator versions {sorted(versions)}. "
                 "Re-score them with `python -m claimlens evaluate <run>` before comparing.\n")
    fmt = lambda k, v: "n/a" if v is None else (f"{v:.3f}" if k == "mean_reward" else f"{v * 100:.1f}%")
    base = runs[0]
    L.append("| Metric | " + " | ".join(f"`{r['_dir']}`" for r in runs) + " |")
    L.append("|---|" + "---|" * len(runs))
    for key, name, higher_better in HEADLINE:
        cells = []
        for r in runs:
            v, b = r["summary"].get(key), base["summary"].get(key)
            cell = fmt(key, v)
            if r is not base and v is not None and b is not None and abs(v - b) > 1e-9:
                better = (v > b) == higher_better
                delta = (v - b) if key == "mean_reward" else (v - b) * 100
                unit = "" if key == "mean_reward" else " pts"
                dfmt = f"{abs(delta):.3f}" if key == "mean_reward" else f"{abs(delta):.1f}"
                cell += f" ({'▲' if v > b else '▼'}{dfmt}{unit}{' ✓' if better else ' ✗'})"
            cells.append(cell)
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    if all(r["fairness"].get("counterfactual_sets") for r in runs):
        L.append("| Counterfactual consistency | " + " | ".join(
            fmt("x", r["fairness"]["counterfactual_consistency"]) for r in runs) + " |")
    L.append("\n**Per-field grounded accuracy**\n")
    L.append("| Field | " + " | ".join(f"`{r['_dir']}`" for r in runs) + " |")
    L.append("|---|" + "---|" * len(runs))
    for f in base["summary"]["per_field"]:
        L.append(f"| `{f}` | " + " | ".join(fmt("x", r["summary"]["per_field"][f]["grounded_accuracy"])
                                           for r in runs) + " |")
    L.append("\n**Error types**\n")
    types = sorted({t for r in runs for t in r["summary"]["error_types"]})
    L.append("| Error type | " + " | ".join(f"`{r['_dir']}`" for r in runs) + " |")
    L.append("|---|" + "---|" * len(runs))
    for t in types:
        L.append(f"| {t} | " + " | ".join(str(r["summary"]["error_types"].get(t, 0)) for r in runs) + " |")
    return "\n".join(L) + "\n"
