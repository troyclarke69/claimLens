"""Command line: python -m claimlens <command> [options]

  generate      create the synthetic dataset
  run           run a predictor over a split, then evaluate it
  evaluate      (re-)score an existing run
  compare       compare runs side by side (e.g. baseline vs SFT)
  verify-audit  check a run's audit log has not been tampered with
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="claimlens", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create the synthetic dataset")
    g.add_argument("--out", default="data")
    g.add_argument("--train", type=int, default=200)
    g.add_argument("--val", type=int, default=50)
    g.add_argument("--test", type=int, default=100)
    g.add_argument("--fair-sets", type=int, default=25, help="counterfactual sets for fairness testing")
    g.add_argument("--fair-variants", type=int, default=4, help="name variants per set")
    g.add_argument("--seed", type=int, default=1234)

    r = sub.add_parser("run", help="run a predictor over a split, then evaluate")
    r.add_argument("--data", default="data")
    r.add_argument("--split", default="test", choices=["train", "val", "test", "fairness"])
    r.add_argument("--predictor", default="simulated", choices=["simulated", "hf"])
    r.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Hugging Face model id (hf only)")
    r.add_argument("--adapter", default=None, help="LoRA adapter path (Phase 2)")
    r.add_argument("--load-in-4bit", action="store_true")
    r.add_argument("--dtype", default="float16")
    r.add_argument("--max-pixels", type=int, default=None)
    r.add_argument("--max-new-tokens", type=int, default=900)
    r.add_argument("--prompt-version", default=None, help="default: latest (see prompts.py)")
    r.add_argument("--finish-json", action="store_true", help="hf only: block stopping until JSON is closed")
    r.add_argument("--limit", type=int, default=None, help="only the first N documents")
    r.add_argument("--runs-dir", default="runs")
    r.add_argument("--run-id", default=None, help="reuse an id to resume an interrupted run")
    r.add_argument("--bias-group", default=None, help="simulated only: plant a bias against this name group")
    r.add_argument("--bias-strength", type=float, default=0.0)

    e = sub.add_parser("evaluate", help="(re-)score an existing run")
    e.add_argument("run_dir")
    e.add_argument("--data", default=None)

    c = sub.add_parser("compare", help="compare runs side by side (first run = reference)")
    c.add_argument("run_dirs", nargs="+")
    c.add_argument("--out", default=None, help="also write the table to this markdown file")

    v = sub.add_parser("verify-audit", help="check an audit log's hash chain")
    v.add_argument("run_dir")

    a = p.parse_args(argv)

    if a.cmd == "generate":
        from .generator import generate_dataset
        info = generate_dataset(a.out, a.train, a.val, a.test, a.fair_sets, a.fair_variants, a.seed)
        print(json.dumps(info, indent=2))
        return 0

    if a.cmd == "run":
        from .runner import evaluate_run, run_predictions
        if a.predictor == "simulated":
            from .predictors import SimulatedPredictor
            pred = SimulatedPredictor(bias_group=a.bias_group, bias_strength=a.bias_strength)
        else:
            from .predictors import HFVisionPredictor
            pred = HFVisionPredictor(a.model, max_new_tokens=a.max_new_tokens, dtype=a.dtype,
                                     load_in_4bit=a.load_in_4bit, max_pixels=a.max_pixels, adapter_path=a.adapter,
                                     finish_json=a.finish_json,
                                     **({"prompt_version": a.prompt_version} if a.prompt_version else {}))
        run_dir = run_predictions(pred, a.data, a.split, a.runs_dir, a.run_id, a.limit)
        m = evaluate_run(run_dir)
        _print_summary(run_dir, m)
        return 0

    if a.cmd == "evaluate":
        from .runner import evaluate_run
        m = evaluate_run(a.run_dir, a.data)
        _print_summary(a.run_dir, m)
        return 0

    if a.cmd == "compare":
        from .compare import compare
        table = compare(a.run_dirs)
        print(table)
        if a.out:
            from pathlib import Path
            Path(a.out).write_text(table, encoding="utf-8")
        return 0

    if a.cmd == "verify-audit":
        from pathlib import Path
        from .audit import verify
        ok, msg = verify(Path(a.run_dir) / "audit.jsonl")
        print(msg)
        return 0 if ok else 1


def _print_summary(run_dir, m):
    s, f = m["summary"], m["fairness"]
    pct = lambda x: "n/a" if x is None else f"{x * 100:.1f}%"
    print(f"\nRun: {run_dir}")
    print(f"  field accuracy     {pct(s['field_accuracy'])}")
    print(f"  grounded accuracy  {pct(s['grounded_accuracy'])}")
    print(f"  valid JSON         {pct(s['json_valid_rate'])}")
    print(f"  hallucination rate {pct(s['hallucination_rate'])}")
    print(f"  mean reward        {s['mean_reward']}")
    if f["counterfactual_sets"]:
        print(f"  counterfactual consistency {pct(f['counterfactual_consistency'])}")
    print(f"  audit log          {m['audit_message']}")
    print(f"  report             {run_dir}/report.md")


if __name__ == "__main__":
    sys.exit(main())
