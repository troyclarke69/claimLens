# ClaimLens: a plain-language guide

This guide explains what each piece does, why it exists, and which interview question it helps you answer. Read
it alongside the code. Each section ends with a small **Try this** exercise, because running things teaches faster
than reading about them.

---

## 1. The big picture

We want a model that reads a claim document and returns each key field together with **proof of where it found
it**. We will improve the model in two ways:

- **SFT (supervised fine-tuning), Phase 2.** Show the model thousands of documents with the correct answers and
  nudge its weights so it copies those answers. It works like studying with an answer key.
- **RL (reinforcement learning), Phase 3.** Let the model answer, **score** each answer automatically, and nudge
  it toward the answers that scored higher. We will use a method called GRPO: for each document the model tries
  several answers, and the better-than-average ones are reinforced. It works like practice with a strict marker.

Both methods need two things first:

1. **Data with known correct answers.** That is the generator.
2. **A way to score an answer.** That is the evaluation harness. For RL the score is literally the training
   signal, called the *reward*.

That is why Phase 1 builds data and evaluation before any training.

```
generator ──► data (images + exact labels)
                 │
predictor ──► raw model text ──► audit log ──► evaluation ──► report / metrics / RL reward
                                                    │
                                              error analysis notebook
```

---

## 2. Component by component

### `generator.py`: synthetic documents

**What it does.** It draws fake claim forms and invoices, writing every string itself, so it knows the exact
value and pixel box of every field. It then makes the image look scanned: slight rotation, blur, noise, washed-out
contrast. About a third of claim forms are "handwritten" in blue ink with wobbly letters.

**Why it matters.** You get perfect labels at zero cost and with no real personal data. Real claim data is
private and regulated, and synthetic data lets you build the whole pipeline first.

**Deliberate traps (distractors).** Invoices show line items, a subtotal and tax as well as the total. Claim forms
show a date of birth, a date reported and a deductible. A model that grabs "the first date" or "the biggest
number" gets caught, and the evaluator names the mistake `distractor_confusion`.

**Fairness sets.** The `fairness` split renders the *same* document several times, changing only the claimant's
name (across synthetic name groups and genders). Any change in the other fields is a red flag.

**Interview question it answers:** *"How did you create your training data, and what are its limits?"* (See
section 5.)

> **Try this:** open a few images in `data/test/images/` next to their `labels/*.json`. Find the `layout` entries
> with `"role": "distractor"`.

### `schema.py` and `prompts.py`: the contract

Every field is either `null` (not on the page) or `{value, evidence_text, page, bbox}`. The prompt is built from
the same field descriptions, so the prompt and the schema can't drift apart. The prompt carries a **version and
a hash**, so every result can be traced to the exact instructions that produced it.

### `evaluation.py`: the heart of the project

For each document it:

1. **Parses** the raw text. Invalid JSON loses everything; a malformed field loses only that field.
2. **Normalises** values, so `$1,240.55` equals `1240.55` and `March 14, 2025` equals `2025-03-14`.
3. **Scores each field**: right or wrong, and if wrong, *what kind* of wrong.
4. **Checks the citation** in two ways:
   - *IoU* (intersection over union): how much the cited box overlaps the true box, from 0 to 1. At 0.5 or
     above it counts as a hit.
   - *Evidence supported*: does the text inside the cited box actually contain the stated value? This catches a
     model that gives the right answer but points at the wrong place, which an auditor can't verify.
5. **Computes the reward**: `0.1 × format + 0.6 × accuracy + 0.3 × grounding`, a single number from 0 to 1.

**Why the reward lives here.** If the RL reward and the evaluation metric were separate code, they could drift
apart, and the model would be trained on one thing and judged on another. Keeping them together avoids that.

**Reward hacking** (keep this in mind for Phase 3): RL optimises *exactly* what you reward, including loopholes.
Two examples:

- If returning `null` everywhere scored well, the model would learn to answer nothing. That is why missed fields
  score zero.
- If citations were rewarded without checking their content, the model would learn to draw boxes anywhere.

Designing a reward that can't be gamed is a core skill, and a great interview topic.

> **Try this:** in `tests/test_core.py`, read `test_wrong_citation_lowers_reward`. Then write your own test: what
> reward does an answer get that is all `null`?

### Fairness metrics (also in `evaluation.py`)

There are two complementary views:

- **Group parity:** is accuracy similar across name groups? The claimant *name itself* is reported separately.
  Misreading some names more than others (for example, dropping accents) is a fairness issue in its own right.
- **Counterfactual consistency:** within a set where only the name differs, do the *other* fields stay
  identical? This is the sharper test, because nothing else changed.

> **Try this:** run the simulated predictor on the fairness split with and without `--bias-group west_african
> --bias-strength 0.5`. Watch counterfactual consistency drop and `total_amount` show up as the field that flips.
> You planted a bias and the harness caught it, which is how you know the harness works.

### `audit.py`: tamper-evident log

Each prediction is logged with *everything needed to reproduce it*: model and revision, prompt hash, dataset hash,
image hash, code version and the raw output. Each record includes the hash of the previous one, forming a chain.
Edit or delete any line and `verify-audit` reports exactly where. Scores are recomputed **from the log**, so the
log is the source of truth.

**Interview line:** "Every number in my report can be traced to an immutable record of what the model saw and
said."

> **Try this:** edit one character in `runs/sim_test/audit.jsonl`, then run
> `python -m claimlens verify-audit runs/sim_test`.

### `predictors.py`: the models

- `SimulatedPredictor`: ground truth plus planted errors. It is only for testing the pipeline on your laptop,
  never for results.
- `HFVisionPredictor`: a real open model (default **Qwen2.5-VL-3B-Instruct**) loaded from Hugging Face on a GPU.
  It also converts box coordinates back to the original image size if the model resized the image. In Phase 2 it
  will load your LoRA adapter with `--adapter`.

### `notebooks/02_error_analysis.ipynb`: where the insight comes from

The report tells you *how well*. This notebook tells you *why*:

- accuracy per field
- error types
- the **lift** table (e.g. "poor scans are 15% of fields but 30% of errors")
- citation quality
- fairness charts
- a gallery of real failures with the true box (green) and the predicted box (orange)

Findings here decide what to do in Phase 2. If most errors are distractor confusions, add more invoices with
tricky subtotals to the training data. If they are near misses on poor scans, that points to image quality or
resolution.

---

## 3. Key terms

| Term | Plain meaning |
|---|---|
| **VLM** | Vision-language model: reads images and text, writes text |
| **SFT** | Supervised fine-tuning: train on (input, correct answer) pairs |
| **LoRA** | Train a small add-on (a few million parameters) instead of the whole model. Cheap, fits a free GPU, easy to version and roll back |
| **QLoRA** | LoRA on a model loaded in 4-bit to save memory |
| **RL / GRPO** | Improve the model using a score rather than an answer key. GRPO compares several attempts at the same input and reinforces the better ones |
| **Reward** | The score RL maximises (ours: `compute_reward`) |
| **Reward hacking** | The model finds a way to score well without doing the task |
| **IoU** | Overlap between two boxes, 0 to 1 |
| **Hallucination** | Output not supported by the input, e.g. a value for a field that is not on the page |
| **Counterfactual fairness test** | Change only a sensitive attribute and check whether the output changes |
| **Baseline** | The un-tuned model's score, the "before" number |

---

## 4. Your first session (about 30 minutes, laptop)

1. Set up and run the quick start in `README.md`.
2. Open three images and their labels, and find the distractors.
3. Read `runs/sim_test/report.md` top to bottom.
4. Run `02_error_analysis.ipynb` on `sim_test`, then on `sim_fair_bias`.
5. Do the tamper test on the audit log.
6. Then go to Kaggle and run the real baseline (`01_kaggle_baseline.ipynb`) with `LIMIT = 10` first.

---

## 5. Honest limitations (say these before an interviewer does)

- **Synthetic is not real.** Real claims have multi-page PDFs, faxes, stamps, real handwriting and messy layouts.
  The synthetic set tests the *method*. A next step would be adding public document sets (FUNSD, CORD, SROIE) and
  a small hand-labelled real sample.
- **Name groups are a crude proxy.** They are useful for audit-style testing, but they are not demographics.
  Real fairness work also needs outcome-level analysis and domain and legal input.
- **Single page, fixed templates.** A fine-tuned model may overfit these templates. That is why the error analysis
  and a held-out template or format are planned.
- **Tight boxes plus an IoU threshold** can be harsh on a model that draws slightly looser boxes. The
  *evidence-supported* metric exists as a more lenient cross-check.
- **The real-model notebook** uses the standard Hugging Face API, but library versions change. If something
  errors on Kaggle, the fix is usually a version pin or a `dtype` or 4-bit setting.

---

## 6. How to describe this truthfully

> "To go deeper on fine-tuning, I built a small end-to-end project on claims documents: a synthetic data
> generator with exact labels and counterfactual fairness sets, an evaluation harness that also serves as the RL
> reward, a hash-chained audit log, and a baseline with an open 3B vision-language model. Next I'm doing LoRA SFT
> and then GRPO on a free GPU, and comparing against a hosted model on accuracy, cost and latency."

Only quote numbers you actually got, and be ready to show the error analysis behind them.

---

## 7. Phase 2: supervised fine-tuning, explained

### What actually happens during training

For each training document, the model gets the image plus the v2 prompt, and we show it **the exact answer we
want**: every field with its value, its text as printed, and a box around the *value* (never the label). Absent
fields are shown as `null`. The model predicts that answer one token at a time; wherever it would have predicted
something different, the **loss** measures how wrong it was, and the optimiser nudges the weights to make the
right token more likely next time. After a few hundred documents, twice over, it has learned the format, the
distractor traps and where to point.

### LoRA in one picture

The model has about 3 billion weights. Instead of changing them, LoRA adds small side matrices to chosen layers
and trains only those (about 30 million numbers, roughly 1%). Why that matters beyond "it fits on a free GPU":

- **The adapter is a separate file (about 120 MB),** so it can be versioned, fingerprinted, audited and rolled
  back independently of the base model.
- **Several adapters can share one base model,** for example one per line of business or per carrier.
- **Rollback is instant:** unload the adapter and you're back to the audited baseline.

We put LoRA on the **language** layers only and keep the **vision encoder frozen**. The Phase 1 errors were
about *what to write and where to point*, not about seeing pixels, and freezing vision also protects the model's
general image understanding.

### The choices, and the Phase 1 finding behind each

| Choice | Why |
|---|---|
| Targets box the **value**, not the label | 59–65% of correct values cited the wrong place, usually the label |
| Absent fields are explicit `null` | Every hallucination was the invoice number copied into a missing claim number |
| Poor scans oversampled to about 35% | They were about 20% of fields but about 49% of errors |
| Same prompt function at training and evaluation time | Training and inference must see identical text |
| Loss only on answer tokens | We teach the answer, not how to repeat the prompt |
| Same test documents (hash-checked) | The before/after comparison must be like-for-like |

### Key hyper-parameters

| Setting | Value | Plain meaning |
|---|---|---|
| `EPOCHS` | 2 | Passes over the training set. More can mean memorising |
| `LEARNING_RATE` | 1e-4 | Step size of each nudge. Too high is unstable, too low learns nothing |
| `LORA_R` | 16 | Size of the adapter matrices: capacity versus overfitting risk |
| `GRAD_ACCUM` | 8 | Average 8 documents before each weight update: smoother learning at batch-size-1 memory cost |

### Reading the training curve

- **Train loss falls and val loss falls:** it's learning.
- **Train loss falls but val loss rises:** it's overfitting, memorising the training documents. Use fewer epochs
  or a lower learning rate.
- **Loss is flat:** the learning rate is too low, or something is wrong with the data.

### What to watch in the results (and what to be honest about)

- **Did grounded accuracy rise sharply?** That's the main goal.
- **Did `distractor_for_absent` (invoice number as claim number) fall to near zero?**
- **Did near-misses fall on poor scans?**
- **Did counterfactual consistency move toward 100%?**
- **Did anything get worse?** Fine-tuning can trade one thing for another. Check every row of the comparison, not
  just the headline.
- **Caveat:** the training and test documents come from the *same templates*. A big jump partly reflects
  learning these specific layouts. The honest next test is a **held-out template** the model never saw in
  training, which tells you how much of the gain generalises.

### Why the evaluator changed to 1.1 (and why that's fine)

Phase 1 inspection showed `POL. 214-4955 F` marked wrong against `POL-214-4955-F`, even though it's the same
ID. Changing scoring rules *after* seeing results is acceptable only if you version the rules, document why, and
re-score every run, baseline included, so the comparison stays fair. Scores are recomputed from the audit logs,
so that's one command per run and needs no GPU.
