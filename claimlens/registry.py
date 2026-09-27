"""Model registry and release gate.

`models/registry.json` is built from what the project already records:
every adapter's training card (lineage + fingerprint) and every run's
metrics.json (evaluation results). Nothing is typed in by hand.

Promoting a model to production goes through a **release gate**: the candidate
must not be worse than the current production model on the checks below, on
the same evaluation sets. Every promotion (and every forced override) is
appended to the registry's changelog -- who/when/why is part of the audit trail.
"""

from __future__ import annotations

import getpass
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REGISTRY_PATH = Path("models/registry.json")

# (metric, higher_is_better, allowed slack in absolute terms) per evaluation split
GATE = {
    "test": [("field_accuracy", True, 0.005), ("grounded_accuracy", True, 0.01),
             ("hallucination_rate", False, 0.005)],
    "holdout": [("field_accuracy", True, 0.01), ("grounded_accuracy", True, 0.02),
                ("hallucination_rate", False, 0.01)],
}
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _run_model_name(run_id: str) -> Optional[str]:
    # run ids look like  <model>_v2_greedy_<split>
    if "_v2_greedy_" not in run_id or run_id.startswith("quick_"):
        return None
    return run_id.split("_v2_greedy_")[0]


def build(adapters_dir: str | Path = "adapters", runs_dir: str | Path = "runs",
          path: Path = REGISTRY_PATH) -> dict:
    old = _load(path)
    models: dict[str, dict] = {"baseline": {"base_model": BASE_MODEL, "adapters": [], "trained_by": None}}

    cards = {}
    for card_path in sorted(Path(adapters_dir).glob("*/training_card.json")):
        card = _load(card_path)
        name = card.get("adapter_name", card_path.parent.name)
        cards[name] = card
    for name, card in cards.items():
        chain = []
        parent = card.get("parent_adapter")
        if parent:  # adapters are stacked: parent first
            chain.append({"name": parent["name"], "sha256": parent["sha256"]})
        chain.append({"name": name, "sha256": card.get("adapter_sha256")})
        models[name] = {"base_model": card.get("base_model", BASE_MODEL), "base_revision": card.get("base_revision"),
                        "adapters": chain, "trained_by": card.get("mode", "sft"),
                        "quick_test": card.get("training", {}).get("quick_test", card.get("quick_test"))}

    for mpath in sorted(Path(runs_dir).glob("*/metrics.json")):
        m = _load(mpath)
        name = _run_model_name(m.get("run_id", ""))
        if not name:
            continue
        name = {"baseline": "baseline"}.get(name, name)
        entry = models.setdefault(name, {"base_model": BASE_MODEL, "adapters": [], "trained_by": "unknown"})
        s = m["summary"]
        entry.setdefault("evaluations", {})[m["split"]] = {
            "run_id": m["run_id"], "evaluator_version": m.get("evaluator_version", "1.0"),
            **{k: s.get(k) for k in ("field_accuracy", "grounded_accuracy", "hallucination_rate", "miss_rate",
                                     "doc_exact_match", "mean_reward", "latency_ms_p50")},
        }
        if m["fairness"].get("counterfactual_sets"):
            entry["evaluations"][m["split"]]["counterfactual_consistency"] = m["fairness"]["counterfactual_consistency"]

    reg = {"production": old.get("production"), "models": models, "changelog": old.get("changelog", []),
           "built_utc": datetime.now(timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return reg


def gate(reg: dict, candidate: str, current: Optional[str]) -> tuple[bool, list[dict]]:
    """Compare candidate vs current production on every gated metric."""
    cand = reg["models"][candidate].get("evaluations", {})
    cur = reg["models"].get(current, {}).get("evaluations", {}) if current else {}
    rows, ok = [], True
    for split, checks in GATE.items():
        if split not in cand:
            rows.append({"split": split, "metric": "(missing)", "result": "FAIL: candidate not evaluated on this split"})
            ok = False
            continue
        if cand[split].get("evaluator_version") != cur.get(split, {}).get("evaluator_version", cand[split].get("evaluator_version")):
            rows.append({"split": split, "metric": "(evaluator)", "result": "FAIL: scored with different evaluator versions"})
            ok = False
        for metric, higher, slack in checks:
            c = cand[split].get(metric)
            p = cur.get(split, {}).get(metric)
            if p is None:
                rows.append({"split": split, "metric": metric, "candidate": c, "production": None, "result": "pass (no production yet)"})
                continue
            passed = (c >= p - slack) if higher else (c <= p + slack)
            ok &= passed
            rows.append({"split": split, "metric": metric, "candidate": c, "production": p,
                         "result": "pass" if passed else "FAIL"})
    if reg["models"][candidate].get("quick_test"):
        rows.append({"split": "-", "metric": "(training)", "result": "FAIL: adapter came from a quick test run"})
        ok = False
    return ok, rows


def promote(candidate: str, reason: str, force: bool = False, path: Path = REGISTRY_PATH) -> tuple[bool, list[dict]]:
    reg = _load(path)
    if candidate not in reg.get("models", {}):
        raise SystemExit(f"unknown model '{candidate}'. Run `python -m claimlens registry build` first.")
    ok, rows = gate(reg, candidate, reg.get("production"))
    if ok or force:
        reg["changelog"].append({
            "utc": datetime.now(timezone.utc).isoformat(), "user": getpass.getuser(),
            "from": reg.get("production"), "to": candidate, "reason": reason,
            "gate_passed": ok, "forced": bool(force and not ok), "gate": rows,
        })
        reg["production"] = candidate
        path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return ok, rows
