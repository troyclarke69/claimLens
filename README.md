# ClaimLens

**Evidence-cited extraction from insurance claim documents, with fine-tuning (SFT), reinforcement learning
(GRPO), fairness testing and a tamper-evident audit trail, built end to end on a free GPU.**

ClaimLens reads a scanned claim form or provider invoice and returns structured JSON in which every field carries
its evidence: the text it read and the pixel box it read it from. The project answers, with measured numbers, the
questions that matter when you adapt a vision-language model for a regulated domain:

- Did fine-tuning help, and does the gain survive document layouts it has never seen?
- Does the model cite real evidence, or invent answers?
- Does its output change when only the claimant's name changes?
- What does reinforcement learning add beyond supervised fine-tuning, when compared fairly?
- Can every prediction be traced, re-scored and audited?

> All data is synthetic. There is no real personal information anywhere.
> **For the one-page summary, see [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md).**

## Results at a glance

Model: Qwen2.5-VL-3B-Instruct with LoRA adapters, on a Kaggle T4. Evaluation: 100 test documents (layouts seen in
training), 100 **held-out** documents (two layouts and label wordings never seen in training), and 100 fairness
documents (25 sets × 4 claimant names). All scored with evaluator 1.1.

| Model | Test: grounded accuracy | Test: hallucination | Held-out: field accuracy | Held-out: grounded | Held-out: exact match | Held-out: hallucination | Held-out: miss |
|---|---|---|---|---|---|---|---|
| Baseline (prompt only) | 37.6% | 7.1% | 90.1% | 38.2% | 53% | 8.7% | 3.7% |
| SFT v1 (300 docs, 2 layouts) | 96.1% | 0% | 89.9% | 84.4% | 49% | 4.9% | 5.2% |
| **SFT v2** (+80 randomised-layout docs) | **96.1%** | **0%** | **92.9%** | **86.4%** | **63%** | 2.9% | **3.9%** |
| GRPO v1 (same 80 docs, cost-weighted reward) | 95.9% | 0% | 89.9% | 82.9% | 45% | **1.9%** | 6.7% |

**Grounded accuracy** means the right value *and* a citation box on the right spot. **Counterfactual
consistency** (the same document with only the name changed) went from 92% for the baseline to 100% after SFT.

### Key findings

1. **Prompting has limits.** A better prompt took valid JSON from 50% to 100%. But the base model still copied
   the invoice number into a missing claim number on 100% of such invoices, despite an explicit "not an invoice
   number" instruction. Labelled examples (SFT) removed that behaviour entirely.
2. **SFT fixed behaviour, not perception.** Grounding went from 38% to 96% and hallucinations to 0%, but
   single-character misreads of long IDs only fell by about a third. The vision encoder was frozen.
3. **Held-out layouts exposed shortcut learning.** About 80% of the grounding gain transferred, but the model had
   learned *label wording*: under "Our reference", claim-number misses rose from 9 to 16 of 50. Training on
   randomised layouts and synonyms fixed most of this (exact match 49% → 63%).
4. **With a control run, RL's contribution shrank to a trade-off.** GRPO cut hallucinations further (1.9%) but
   missed more. SFT on the *same* documents beat it on almost everything. Here, data diversity drove
   generalisation, and RL moved the caution/coverage trade-off, which is what its cost-weighted reward encoded.
5. **The fairness test found a robustness issue, not bias.** The baseline's name-induced changes were unstable
   readings of long IDs, not a consistent disadvantage for any group. That distinction matters.

Full tables and analysis: [`docs/RESULTS.md`](docs/RESULTS.md). Cost and deployment trade-offs:
[`docs/BUILD_VS_BUY.md`](docs/BUILD_VS_BUY.md).

## How it's built

```
generator ─► synthetic documents + exact labels (values, boxes, distractors, name-swapped fairness sets,
             held-out and randomised layouts)
                │
predictor ─► raw model text ─► hash-chained audit log ─► evaluator (versioned) ─► reports · comparisons
 (baseline /                                                 │                     · RL reward
  LoRA SFT /                                                 └─► error analysis · fairness · release gate
  GRPO / hosted API)                                                                   │
                                                             registry ◄────────────────┘
                                                                │
                                                FastAPI service (review flags + audit receipts) · Docker
```

## Quick start (laptop, CPU only)

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt -r requirements-service.txt

python -m pytest -q                                       # 31 tests
python -m claimlens generate --out data                   # synthetic dataset
python -m claimlens run --split test --run-id sim_test    # pipeline check with the simulated predictor
# needs run outputs from the Kaggle notebooks:
python -m claimlens compare runs/sft_v1_v2_greedy_holdout runs/sft_v2ctrl_v2_greedy_holdout runs/grpo_v1_v2_greedy_holdout
```

The **simulated predictor** is not a model: it copies the ground truth and injects known errors at known rates,
to prove the evaluator and the fairness checks catch them. Real model results come from the Kaggle notebooks.

### Run the service

```bash
uvicorn claimlens.service:app --port 8000                 # then open http://localhost:8000/docs
# or
docker build -t claimlens . && docker run -p 8000:8000 -v "%cd%/data:/app/data" claimlens
```

`POST /v1/extract` takes a document image and returns the fields with their evidence, **review flags**
(unparseable output, a required field missing, a value without a citation, or a value that disagrees with its own
evidence text) and an **audit receipt** (the hash of its tamper-evident log record). On a laptop the service runs
the simulated predictor. With a GPU, set `CLAIMLENS_PREDICTOR=hf` to serve the registry's production model with
its adapter fingerprints verified at load.

### Model registry and release gate

```bash
python -m claimlens registry build                          # from training cards + run metrics
python -m claimlens registry promote sft_v1 --reason "first fine-tuned model"
python -m claimlens registry promote sft_v2ctrl --reason "better held-out generalisation"
```

A candidate is promoted only if it isn't worse than production on accuracy, grounding and hallucination rate, on
both the test and held-out sets, scored with the same evaluator. Overrides need `--force` and are recorded in the
changelog.

## Reproducing on a free GPU (Kaggle)

| Notebook | What it does | Time on a T4 |
|---|---|---|
| `01_kaggle_baseline.ipynb` | Un-tuned baseline, prompt experiments | ~2 h |
| `02_error_analysis.ipynb` | (laptop) why the model fails: fields, error types, slices, fairness, failure gallery | minutes |
| `03_kaggle_sft.ipynb` | LoRA SFT v1 + identical re-evaluation | ~4.5 h |
| `04_kaggle_holdout.ipynb` | Baseline vs SFT on never-seen layouts | ~1.75 h |
| `05_kaggle_phase3.ipynb` | `MODE="control"` (SFT v2) or `MODE="grpo"` | ~2 h / ~4 h |

Each notebook has a `QUICK_TEST` dry-run mode. Always dry-run first.

## Project layout

```
claimlens/
  schema.py       output contract: every field = value + evidence_text + page + bbox, or null
  generator.py    synthetic documents: training, held-out and randomised layouts; fairness sets
  names.py        synthetic name pools for counterfactual fairness sets
  normalize.py    '$1,240.55' == '1240.55'; 'March 14, 2025' == '2025-03-14'
  evaluation.py   scoring, error taxonomy, fairness metrics, RL rewards (versioned)   <- the core
  audit.py        append-only, hash-chained audit log + verifier
  prompts.py      versioned, hashed prompts (v1 kept for reproducibility)
  predictors.py   simulated, Hugging Face (stacked LoRA adapters, fingerprint-checked), hosted API
  runner.py       run a predictor over a split, log everything, score it
  sft.py          LoRA SFT: examples, masking, training loop, training cards, fingerprints
  grpo.py         GRPO: group sampling, advantages, KL to reference, reward-hacking monitors
  compare.py      side-by-side comparison of runs
  costs.py        cost per document from logged latency or tokens
  registry.py     model registry + release gate + changelog
  service.py      FastAPI service with review flags and audit receipts
notebooks/        Kaggle notebooks (see table above)
docs/             RESULTS.md · BUILD_VS_BUY.md
tests/            31 tests (evaluation, generator, audit, SFT/GRPO pieces, service, registry, costs)
GUIDE.md          plain-language explanation of every piece and design choice
PROJECT_SUMMARY.md  one-page summary
Dockerfile        CPU demo image (GPU build arg documented inside)
```

## Auditability and reproducibility

- **Every prediction** is logged with the model revision, adapter fingerprints, prompt version and hash, dataset
  hash, image SHA-256, code version, latency and raw output, in a **hash chain**.
  `python -m claimlens verify-audit <run>` pinpoints any edited or deleted line.
- **Scores are recomputed from the audit log**, so the log is the source of truth.
- **Scoring rules are versioned.** When they changed (1.0 → 1.1), every run was re-scored, and the old scores were
  kept as `metrics_eval-1.0.json`.
- **Every adapter** has a training card (base-model revision, parent adapter, data hashes, hyper-parameters) and a
  weight fingerprint, and is verified before evaluation or serving.
- **Test sets are hash-checked** across every phase, so all comparisons are on identical documents.

## Limitations

Synthetic single-page documents from a small number of templates. 100 documents per evaluation set. Single runs
(no repeated seeds). Name groups are a crude, synthetic proxy for audit-style testing, not demographics. The hosted
model comparison is set up but not yet run (`docs/BUILD_VS_BUY.md`). A production system would need real, consented
data, multi-page and handwritten documents, larger evaluation sets with confidence intervals, and legal and domain
review of fairness criteria.
