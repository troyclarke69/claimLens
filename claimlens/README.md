# ClaimLens

**Evidence-cited extraction from insurance claim documents: a hands-on SFT + RL learning project.**

ClaimLens reads a scanned claim form or provider invoice and returns structured JSON. Every field carries its
evidence: the text it read and the pixel box where it read it. The project is built to answer, with real
numbers, the questions that matter when you fine-tune AI for a regulated domain:

- Did fine-tuning actually help, and on which fields?
- Does the model cite real evidence, or does it invent answers?
- Does it treat claimants differently depending on their name?
- Can every prediction be traced back and audited?

> **Status: Phase 1 complete (real baseline measured); Phase 2 (SFT) built and ready to run.** All data is
> synthetic, and there is no real personal information anywhere.

## Roadmap

| Phase | Contents | Status |
|---|---|---|
| **1** | Synthetic data · evaluation harness (incl. fairness) · audit log · baseline model · error analysis | ✅ done |
| **2** | LoRA SFT of Qwen2.5-VL-3B on Kaggle (free T4) · before/after comparison | 🔧 built, ready to run |
| 3 | GRPO (RL) using the evaluation harness as the reward · reward-hacking checks | |
| 4 | FastAPI + Docker service · model versioning · build-vs-buy comparison vs a hosted model | |
| 5 | Write-up: architecture, results, lessons, what changes in production | |

## Quick start (laptop, CPU only, about 2 minutes)

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python -m pytest -q                                    # 23 tests
python -m claimlens generate --out data                # 200 train / 50 val / 100 test / 100 fairness docs
python -m claimlens run --split test --run-id sim_test # simulated predictor: tests the pipeline
python -m claimlens run --split fairness --run-id sim_fair_bias --bias-group west_african --bias-strength 0.5
python -m claimlens verify-audit runs/sim_test
jupyter notebook notebooks/02_error_analysis.ipynb
```

The **simulated predictor** is not a model. It copies the ground truth and injects known errors at known rates,
so you can check that the harness catches every kind of mistake, and that the fairness report catches a bias you
planted on purpose. Real model numbers come from the Kaggle notebook.

## Baseline on a real model (free Kaggle GPU)

Upload `notebooks/01_kaggle_baseline.ipynb` to Kaggle, set the accelerator to T4 and turn Internet on, then run
it. The notebook explains each step. It produces `runs/baseline_qwen25vl3b_test/report.md`, the "before" number
for Phase 2.

## Phase 1 results: un-tuned Qwen2.5-VL-3B baseline

Kaggle T4, prompt v2, greedy decoding, 100 test and 100 fairness documents. Scored with evaluator 1.0; re-score
with 1.1 before comparing against Phase 2.

| Metric | Test | Fairness split |
|---|---|---|
| Valid JSON | 100% | 100% |
| Field accuracy | 92.7% | 93.1% |
| Grounded accuracy (right value **and** right location) | 37.6% | 32.9% |
| Hallucination rate | 7.1% | 12.1% |
| Document exact match | 64% | 58% |
| Counterfactual consistency | n/a | 88% |
| Mean reward | 0.803 | 0.801 |
| Latency p50 | ~30 s | ~30 s |

**What the error analysis showed:**

1. **Prompting got us part of the way.** Prompt v1 gave only 50% valid JSON, because the model stopped
   mid-answer. Prompt v2 (compact one-line fields, every key shown) reached 100%. Constrained decoding
   ("finish JSON") added nothing on top, so it stays off.
2. **The model reads well but can't show its work.** 59–65% of *correct* values cite the wrong location, usually
   the label ("Invoice #:") instead of the value.
3. **Hallucinations are one specific behaviour.** Every hallucination was the invoice number copied into
   `claim_number` on invoices with no claim reference, even though the prompt explicitly says "not an invoice
   number". That's a clear case for fine-tuning over further prompt changes.
4. **Errors concentrate.** Poor scans are about 20% of fields but about 49% of errors, and invoices about 49% of
   fields but 73% of errors. Most errors are single-character misreads of long IDs.
5. **Fairness.** In 3 of 25 counterfactual sets, changing only the name changed another field. Inspection showed
   unstable reading of long IDs, not a consistent disadvantage for any group. It's a robustness finding, surfaced
   by the fairness test. Group sizes (12–25 documents) are too small to size group gaps reliably.

## Phase 2: supervised fine-tuning (SFT)

`notebooks/03_kaggle_sft.ipynb` trains a LoRA adapter (language model only, vision encoder frozen) on the 300
training documents, with poor scans oversampled to about 35%. It then re-runs the identical test and fairness
evaluation. Do a 10-minute dry run first (`QUICK_TEST = True`), then the full run in the background (about 3
hours).

Compare afterwards on your laptop:

```bash
python -m claimlens evaluate runs/baseline_v2_greedy_test        # re-score baseline with evaluator 1.1
python -m claimlens evaluate runs/baseline_v2_greedy_fairness
python -m claimlens compare runs/baseline_v2_greedy_test runs/sft_v1_v2_greedy_test --out phase2_test.md
python -m claimlens compare runs/baseline_v2_greedy_fairness runs/sft_v1_v2_greedy_fairness --out phase2_fairness.md
```

## Project layout

```
claimlens/
  schema.py       output contract: every field = value + evidence_text + page + bbox, or null
  generator.py    synthetic claim forms & invoices with exact labels, scan noise, skew, "handwriting"
  names.py        synthetic name pools for counterfactual fairness sets
  normalize.py    '$1,240.55' == '1240.55'; 'March 14, 2025' == '2025-03-14'
  evaluation.py   scoring, error classification, RL reward, fairness metrics   <- the core
  audit.py        append-only, hash-chained audit log + verifier
  prompts.py      versioned, hashed extraction prompt
  predictors.py   SimulatedPredictor (CPU) and HFVisionPredictor (GPU, any HF vision-language model)
  runner.py       run a predictor over a split, log everything, score it
  sft.py          Phase 2: training examples, LoRA setup, training loop, adapter fingerprint + training card
  compare.py      side-by-side comparison of runs
  report.py       markdown report per run
notebooks/
  01_kaggle_baseline.ipynb   real-model baseline on a free Kaggle T4
  02_error_analysis.ipynb    why the model fails: fields, error types, slices, fairness, picture gallery
  03_kaggle_sft.ipynb        Phase 2: LoRA fine-tune + identical re-evaluation on a free Kaggle T4
tests/            pytest suite
GUIDE.md          plain-language explanation of every piece and design choice
```

## What gets measured

| Metric | Meaning |
|---|---|
| Field accuracy | Correct value (after normalisation). Returning `null` for an absent field counts as correct |
| Grounded accuracy | Correct value **and** the cited box overlaps the true location (IoU ≥ 0.5) |
| Evidence supported | The cited box really contains the stated value, which catches made-up citations |
| Hallucination rate | Invented a value for a field that is not on the page |
| Error types | `near_miss`, `distractor_confusion` (subtotal read as total), `distractor_for_absent` (e.g. invoice number given as a missing claim number), `field_swap`, `diacritics`, `missed`, `hallucinated`, `unparseable` |
| Fairness | Accuracy by synthetic name group and gender, plus **counterfactual consistency**: same document with only the name changed, do the other fields stay identical? |
| RL reward | `0.1·format + 0.6·accuracy + 0.3·grounding`, in [0, 1]. Phase 3 optimises exactly this |

## Evaluator versions

Scoring rules are versioned (`EVALUATOR_VERSION` in `evaluation.py`). When they change, every run is re-scored
from its audit log (no GPU needed), and the previous scores are kept as `metrics_eval-<old>.json`.

- **1.0:** initial rules.
- **1.1:** IDs are compared on letters and digits only (`POL. 214-4955 F` equals `POL-214-4955-F`), and
  `distractor_for_absent` is split out of `hallucinated`.

## Auditability

Each prediction is logged with the model ID and revision, the prompt version and hash, the dataset hash, the
image SHA-256, the code version, the latency and the raw output. Records are hash-chained, so
`python -m claimlens verify-audit <run>` reports the exact line where a record was edited or removed. Scores are
always recomputed from the audit log, so the log is the single source of truth.
