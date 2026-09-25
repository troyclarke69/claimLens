"""Markdown report for one run."""

from __future__ import annotations

from pathlib import Path


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def write_report(path: Path, config: dict, m: dict):
    s, fr = m["summary"], m["fairness"]
    L = []
    L.append(f"# ClaimLens evaluation report — `{config['run_id']}`\n")
    if config["model_id"] == "simulated-predictor":
        L.append("> **Simulated predictor.** These numbers come from ground truth with planted errors, used to "
                 "test the pipeline. They are NOT model results.\n")
    L.append(f"- **Model:** `{config['model_id']}` (revision `{config['model_revision']}`)")
    L.append(f"- **Prompt:** {config['prompt_version']} (hash `{config['prompt_hash']}`), "
             f"decoding: {config.get('decoding', 'n/a')}")
    L.append(f"- **Data:** split `{config['split']}`, dataset hash `{config['dataset_hash'][:16]}…`, "
             f"{s['n_docs']} documents")
    L.append(f"- **Code version:** `{config['code_version']}` · **evaluator** {m.get('evaluator_version', '1.0')}")
    L.append(f"- **Audit log:** {'✅' if m['audit_chain_ok'] else '❌'} {m['audit_message']}\n")

    L.append("## Headline metrics\n")
    L.append("| Metric | Value | What it means |")
    L.append("|---|---|---|")
    rows = [
        ("Valid JSON", s["json_valid_rate"], "Output could be parsed at all"),
        ("Schema-valid", s["schema_valid_rate"], "Output matched the required structure exactly"),
        ("Field accuracy", s["field_accuracy"], "Fields with the right value (incl. correctly-null fields)"),
        ("Grounded accuracy", s["grounded_accuracy"], "Right value AND citation box on the right spot (IoU ≥ 0.5)"),
        ("Evidence supported", s["evidence_supported_rate"], "Cited box really contains the stated value"),
        ("Hallucination rate", s["hallucination_rate"], "Gave a value for a field NOT on the page, invented or copied from elsewhere (lower is better)"),
        ("Miss rate", s["miss_rate"], "Returned null for a field that IS on the page (lower is better)"),
        ("Document exact match", s["doc_exact_match"], "Every field correct"),
    ]
    for name, v, desc in rows:
        L.append(f"| {name} | {_pct(v)} | {desc} |")
    L.append(f"| Mean RL reward | {s['mean_reward']:.3f} | The number Phase 3 (GRPO) will optimise |"
             if s["mean_reward"] is not None else "| Mean RL reward | n/a | |")
    if "latency_ms_p50" in s:
        L.append(f"| Latency p50 / p95 | {s['latency_ms_p50']:.0f} / {s['latency_ms_p95']:.0f} ms | Per document |")

    L.append("\n## Per field\n")
    L.append("| Field | Accuracy | Grounded accuracy | Docs where present |")
    L.append("|---|---|---|---|")
    for f, v in s["per_field"].items():
        L.append(f"| `{f}` | {_pct(v['accuracy'])} | {_pct(v['grounded_accuracy'])} | {v['n_present']} |")

    L.append("\n## Slices (field accuracy)\n")
    for title, key in [("Document type", "by_doc_type"), ("Scan quality", "by_quality"),
                       ("Handwritten", "by_handwritten"), ("Line of business", "by_line_of_business")]:
        L.append(f"**{title}:** " + " · ".join(f"{k}: {_pct(v['field_accuracy'])}" for k, v in s[key].items()))
        L.append("")

    L.append("## Error types\n")
    if s["error_types"]:
        L.append("| Error type | Count |")
        L.append("|---|---|")
        for k, v in s["error_types"].items():
            L.append(f"| {k} | {v} |")
    else:
        L.append("No errors.")

    L.append("\n## Fairness\n")
    L.append("Name groups are synthetic proxies used for audit-style testing, not real demographics.\n")
    L.append("| Group | Docs | Other-fields accuracy | Name accuracy |")
    L.append("|---|---|---|---|")
    for g, v in fr["by_group"].items():
        L.append(f"| {g} | {v['n_docs']} | {_pct(v['other_fields_accuracy'])} | {_pct(v['name_accuracy'])} |")
    L.append("")
    L.append(f"- Largest gap between groups — other fields: **{_pct(fr['group_gap_other_fields'])}**, "
             f"claimant name: **{_pct(fr['group_gap_name'])}**")
    L.append(f"- Gender gap — other fields: **{_pct(fr['gender_gap_other_fields'])}**, "
             f"claimant name: **{_pct(fr['gender_gap_name'])}**")
    if fr["counterfactual_sets"]:
        L.append(f"- **Counterfactual consistency:** {_pct(fr['counterfactual_consistency'])} of "
                 f"{fr['counterfactual_sets']} sets gave identical non-name fields when only the name changed")
        flips = {k: v for k, v in fr["counterfactual_flip_rate_by_field"].items() if v}
        if flips:
            L.append("- Fields that changed with the name: " + ", ".join(f"`{k}` ({_pct(v)})" for k, v in flips.items()))
    else:
        L.append("- Counterfactual consistency: run on the `fairness` split to measure this.")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
