# ClaimLens: detailed results by phase

All runs: Kaggle T4, Qwen2.5-VL-3B-Instruct, prompt v2, greedy decoding. All runs are scored with evaluator 1.1 unless noted.

## Phase 1 results: un-tuned Qwen2.5-VL-3B baseline

Kaggle T4, prompt v2, greedy decoding, 100 test and 100 fairness documents. The table shows the original scoring
(evaluator 1.0). Re-scored with 1.1: field accuracy 93.0% / 93.4%, and counterfactual consistency 92% (one
"flip" was only punctuation). Phase 2 comparisons use the 1.1 numbers.

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

## Phase 2 results: LoRA SFT

Same 100 test and 100 fairness documents (hash-checked), same prompt, evaluator 1.1 for both runs.

| Metric | Baseline (test / fairness) | **SFT** (test / fairness) |
|---|---|---|
| Grounded accuracy | 37.6% / 32.9% | **96.1% / 96.8%** |
| Hallucination rate | 7.1% / 12.1% | **0% / 0%** |
| Field accuracy | 93.0% / 93.4% | **97.0% / 97.4%** |
| Document exact match | 64% / 58% | **84% / 84%** |
| Counterfactual consistency | 92% | **100%** |
| Mean reward | 0.805 / 0.802 | **0.970 / 0.975** |

**What SFT fixed:** behaviour. Boxes now land on values, not labels. The invoice-number-as-claim-number errors
(`distractor_for_absent`) went from 6 and 16 to 0, and subtotal/report-date confusions went from 4 and 4 to 0.
Changing only the claimant's name no longer changes any other field.

**What it only partly fixed:** perception. Single-character misreads (`near_miss`) fell only from 30 to 19 and
from 22 to 17, and `claim_number` is still the weakest field (about 89–90% grounded). That's consistent with
training only the language layers while the vision encoder stayed frozen.

**Caveat:** training and test documents share the same two templates. See the held-out test below.

## Held-out template test

`notebooks/04_kaggle_holdout.ipynb` generates 100 documents in two layouts **never used in training** (a boxed
"Notice of Loss" and a "Statement of Account"), with different positions and different wording ("Insured",
"Our reference", "Balance due"). It then evaluates both the baseline and the SFT adapter on them. The adapter's
fingerprint is checked against its training card before use.

```bash
python -m claimlens generate --out data --holdout 100 ...   # the notebook does this for you
python -m claimlens compare runs/baseline_v2_greedy_holdout runs/sft_v1_v2_greedy_holdout --out holdout.md
```

### Held-out results

| Metric | Baseline | SFT v1 |
|---|---|---|
| Grounded accuracy | 38.2% | **84.4%** (vs 96.1% in-distribution) |
| Field accuracy | 90.1% | 89.9% (no gain) |
| Hallucination rate | 8.7% | 4.9% |
| Miss rate | 3.7% | 5.2% |
| Document exact match | 53% | 49% |

**What transferred:** pointing at values (about 80% of the grounding gain) and avoiding distractors (9 → 1).
**What didn't:** the value-accuracy gain. On unfamiliar wording the fine-tuned model **abstains more**. Misses of
`claim_number` under the label "Our reference" rose from 9 to 16 of 50, and of `incident_type` under "Nature of
loss" from 0 to 3. Where the wording was close to training ("Policy no." vs "Policy number"), misses fell from
3 to 0. In short, the model learned label text, not field meaning. Part of it is also a genuinely ambiguous
specification: an insurer's "our reference" usually *is* the claim number, and the schema should say so.

## Phase 3: RL (GRPO) vs an SFT control

`notebooks/05_kaggle_phase3.ipynb`, run twice (`MODE = "control"`, then `MODE = "grpo"`). Both start from the
SFT v1 model, train a fresh LoRA on the **same 80 documents** in *randomised* layouts (shuffled positions,
synonym labels, values beside or below, with none of the held-out wording), and are evaluated on the held-out
templates and the original test set.

- **The control** (`sft_v2ctrl`) imitates the correct answers. It measures what the new data alone gives.
- **GRPO** (`grpo_v1`) samples 4 answers per document, scores them with a **cost-weighted reward**, and
  reinforces the above-average ones, with a KL penalty to stay close to SFT v1:

| Outcome per field | Reward |
|---|---|
| Correct value | +1 (half of it depends on citing the right place) |
| Correctly null | +1 |
| Missed (null, but the value is on the page) | −0.5 |
| Wrong value, or a value for an absent field | −1 |
| Unparseable output | −1 for the whole answer |

The costs encode a business judgement (a confidently wrong claim number is worse than an honest gap), which
imitation learning cannot express. Reward-hacking signals (null answers per sample, box sizes, answer length,
KL) are logged at every step.

```bash
python -m claimlens compare runs/sft_v1_v2_greedy_holdout runs/sft_v2ctrl_v2_greedy_holdout runs/grpo_v1_v2_greedy_holdout --out phase3_holdout.md
python -m claimlens compare runs/sft_v1_v2_greedy_test runs/sft_v2ctrl_v2_greedy_test runs/grpo_v1_v2_greedy_test --out phase3_test.md
```

### Phase 3 results: GRPO vs the SFT control (same 80 randomised-layout documents)

**Held-out templates** (never trained on):

| Metric | SFT v1 (start) | **SFT control** | GRPO |
|---|---|---|---|
| Field accuracy | 89.9% | **92.9%** | 89.9% |
| Grounded accuracy | 84.4% | **86.4%** | 82.9% |
| Document exact match | 49% | **63%** | 45% |
| Hallucination rate | 4.9% | 2.9% | **1.9%** |
| Miss rate | 5.2% | **3.9%** | 6.7% |
| Near-misses / misses (count) | 26 / 24 | **14 / 16** | 24 / 33 |
| Mean reward | 0.892 | **0.917** | 0.888 |

**Original test set:** all three are within about 1 point of each other (field accuracy 97.0 / 97.4 / 97.1%,
grounded 96.1 / 96.1 / 95.9%, hallucinations 0%). There were no regressions.

**Findings:**

1. **The data did the work.** Plain SFT on randomised layouts improved nearly every held-out metric (exact match
   49% → 63%, misses −33%, near-misses −46%).
2. **RL traded coverage for caution.** GRPO had the lowest hallucination rate, which is what the cost-weighted
   reward asked for, but more misses and lower exact match. It moved the model along the caution/coverage
   trade-off rather than improving it.
3. **The control run is what made this visible.** GRPO vs SFT v1 alone reads as "hallucinations −60%". Only the
   control shows that the same data used with SFT achieves most of the benefit without the extra misses.
4. **I checked my own interpretation.** The training log first suggested the model was *learning* to abstain
   (nulls per answer 0.73 → 1.9). Comparing against true nulls per document, the excess stayed flat at about
   +0.1 to +0.2. The rise was document mix, not learned behaviour.
5. **The RL signal was thin.** 39 of 80 GRPO steps had no learning signal (all 4 sampled answers scored the
   same), because the SFT model was already consistent on most documents.
6. **One SFT epoch is enough here.** Control validation loss was 0.0868 after both epoch 1 and epoch 2.

**Caveats:** 100 held-out documents, a single run of each method (no repeated seeds), and a small, untuned RL
setup. The fair claim is "at this scale and budget".
