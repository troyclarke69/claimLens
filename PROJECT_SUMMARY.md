# ClaimLens: project summary

**Fine-tuning a vision-language model for auditable, evidence-cited claims extraction, and testing honestly what
SFT and RL each contribute.**

## The problem

Claims decisions rest on evidence buried in scanned forms and invoices. An extraction system for this domain must
be accurate, and also **show where each answer came from**, avoid inventing values, behave the same regardless of
who the claimant is, and leave an audit trail. I built a small end-to-end system to learn and demonstrate how
supervised fine-tuning (SFT) and reinforcement learning (RL) serve those goals, on a free GPU.

## What I built

- **A synthetic data generator** with exact labels: realistic claim forms and invoices with scan noise, skew,
  "handwriting" and deliberate traps (subtotals, invoice numbers, dates of birth). It also produces
  **counterfactual fairness sets** (identical documents, only the name changed), **held-out layouts** never used
  in training, and **randomised layouts** with synonym labels.
- **An evaluation harness**, the core of the project. It measures value accuracy, **grounding** (is the cited box
  on the right spot?), evidence support, hallucination and misses, uses an error taxonomy, and runs fairness checks.
  The same code produces the RL reward, so training and evaluation can't drift apart. Scoring rules are versioned,
  and all runs were re-scored when they changed.
- **Training:** LoRA SFT of Qwen2.5-VL-3B (language layers only), and **GRPO** with a **cost-weighted reward**
  that encodes a business judgement: a wrong value costs twice what an honest "not found" does. GRPO was compared
  against an **SFT control on identical data**. It includes reward-hacking monitors.
- **Auditability:** a hash-chained audit log of every prediction; training cards and weight fingerprints for every
  adapter, verified before use; and a model registry with a **release gate** and changelog.
- **A service:** a FastAPI endpoint that returns fields with evidence, **human-review flags** and an audit receipt.
  Dockerised.

## Results (Qwen2.5-VL-3B on a T4, 100 documents per set)

| Model | Test: grounded | Test: hallucination | Held-out layouts: grounded | Held-out: exact match | Held-out: hallucination | Held-out: miss |
|---|---|---|---|---|---|---|
| Baseline (prompt only) | 37.6% | 7.1% | 38.2% | 53% | 8.7% | 3.7% |
| SFT v1 | 96.1% | 0% | 84.4% | 49% | 4.9% | 5.2% |
| **SFT v2** (+ randomised layouts) | **96.1%** | **0%** | **86.4%** | **63%** | 2.9% | **3.9%** |
| GRPO (same data as SFT v2) | 95.9% | 0% | 82.9% | 45% | **1.9%** | 6.7% |

Counterfactual consistency (only the name changes; do the other fields stay the same?) went from 92% for the
baseline to **100%** after SFT.

## What I learned (and would say in a design review)

1. **Know when prompting stops working.** Prompt changes fixed the output format (valid JSON 50% → 100%), but the
   model still copied the invoice number into a missing claim number on *every* such invoice, against an explicit
   instruction. That's a behaviour, and behaviours are trained out with labelled examples.
2. **Measure generalisation, not just accuracy.** In-distribution results looked excellent (96% grounded). A
   held-out layout test showed the model had partly learned *label wording*: under "Our reference", claim-number
   misses nearly doubled. Some of that is also an **unclear specification**: whether "our reference" is the claim
   number is a product decision to write into the schema.
3. **Use controls before crediting a method.** GRPO alone looked like "hallucinations −60%". The SFT control on the
   same documents showed that the new *data* drove the generalisation gains, and that RL mainly shifted the
   caution/coverage trade-off, exactly as its reward costs specified. RL's real value here is **encoding business
   costs** that imitation can't express, not adding capability.
4. **Check your own story.** Training logs suggested the RL model was *learning* to abstain. Comparing against
   ground-truth nulls per document showed the trend was document mix. I reported the corrected finding.
5. **Fairness testing can surface robustness bugs.** Name-swap flips in the baseline turned out to be unstable
   digit reading, not group bias. The distinction matters for both the fix and the report.
6. **Metrics are decisions.** Changing one ID-matching rule moved a fairness number by 4 points. That's why
   scoring is versioned and every run is re-scored.

## Build vs buy (see `docs/BUILD_VS_BUY.md`)

At measured, unoptimised throughput, self-hosting costs about $3–4.5 per 1,000 documents on a T4, against about
$2–4 for a small hosted model (Batch or standard API). Cost doesn't decide it. **Data residency, reproducibility
(fingerprinted weights you can re-run for years), and the ability to fine-tune** favour self-hosting the
high-volume path, with a stronger model or a human handling documents the service flags for review.

## Limitations and next steps

The data is synthetic and single-page, with few templates; 100 documents per test set; single runs; and a small RL
budget (about half the GRPO steps had no learning signal). Next I'd: measure a hosted model with the same harness;
add real, consented documents and larger evaluation sets with confidence intervals; adapt the vision encoder or
raise resolution to address digit misreads; serve with a batched inference server; and write the "reference =
claim number" rule into the schema.
