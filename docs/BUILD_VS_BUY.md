# Build vs buy: a self-hosted fine-tuned 3B model vs a hosted frontier API

*Prices checked September 2026 (sources at the end). Self-hosted latency is **measured** in this project;
hosted accuracy is **not yet measured**. See "How to measure the hosted side" below.*

## 1. Cost per document

**Workload:** one page, 980 × 1260 px, the v2 prompt, and about 400 output tokens (measured: the SFT model writes
about 390 tokens).

### Self-hosted: Qwen2.5-VL-3B + LoRA, on one T4

Measured in our runs: **about 26–30 s per document**, unbatched HF `generate` in fp16 on a Kaggle T4.

| T4 price (on demand) | $ per 1,000 documents |
|---|---|
| $0.15/h (cheapest marketplace) | ~$1.1–1.3 |
| $0.42/h (Google Cloud) | ~$3.0–3.5 |
| $0.53/h (AWS / Azure) | ~$3.8–4.4 |

These numbers are the **worst case** for self-hosting. We ran one document at a time with no serving optimisation.
A batched inference server (vLLM, SGLang or TGI with continuous batching) typically gets several times more
throughput from the same GPU. That's an estimate we haven't measured; measuring it is the first step of a real
evaluation.

### Hosted API (Anthropic, same prompt)

Estimated tokens per document: about **1,600 image tokens** (⌈980/28⌉ × ⌈1260/28⌉ = 1,575) + about 650 prompt
tokens ≈ **2,250 input**, plus about **400 output**.

| Model | $/MTok in / out | $ per 1,000 documents | With Batch API (−50%) |
|---|---|---|---|
| Claude Haiku 4.5 | $1 / $5 | ~$4.25 | ~$2.13 |
| Claude Sonnet 5 | $2 / $10 | ~$8.50 | ~$4.25 |
| Claude Opus 5.5 | $4 / $20 | ~$17.00 | ~$8.50 |

### Fixed vs variable cost

An always-on T4 costs about **$387 a month** ($0.53 × 730 h), whether it processes 1 document or 90,000. At our
unoptimised 28 s per document, one T4 handles about **94,000 documents a month**.

- **Break-even against Haiku 4.5:** about 91,000 documents a month, roughly full capacity. Against Haiku 4.5 via
  the Batch API, about 180,000.
- So **on raw cost alone, at this throughput, a small hosted model is competitive.** Self-hosting wins clearly
  only with an optimised server, high and steady volume, or scale-to-zero serverless GPUs.

**Conclusion:** cost doesn't decide this. The factors below do.

## 2. What actually decides it for claims

| Factor | Self-hosted fine-tuned model | Hosted API |
|---|---|---|
| **Sensitive data** (medical and claims records) | Stays inside your cloud account | Leaves your perimeter. Needs contractual data-handling terms, reviewed by legal and compliance |
| **Reproducibility and audit** | Weights and adapter fingerprinted (`4c7f38d9…`). The same model can be re-run in 5 years | Hosted snapshots get deprecated. Re-running an old decision on the *same* model may be impossible |
| **Adapting to your documents** | Proven here: SFT took grounding from 38% to 96%, and randomised-layout SFT improved held-out exact match from 49% to 63% | Prompting and few-shot only (fine-tuning options vary by vendor) |
| **Accuracy out of the box** | A 3B model needed fine-tuning to be useful | Frontier models are likely much stronger zero-shot; **measure it** |
| **Operational burden** | GPUs, serving, scaling, patching, on-call | None beyond the API integration |
| **Latency** | Tunable (batching, bigger GPU) | Typically seconds, subject to rate limits |
| **Fairness and bias testing** | Full control; the counterfactual harness runs on every release | The same harness works, but the model can change underneath you |

## 3. Recommendation (for a system like this)

A **tiered design**, with the gate between tiers driven by the service's existing `review.needs_review` flags:

1. **Tier 1:** the self-hosted fine-tuned model handles the high-volume, well-understood document types. It's
   cheap at scale, fully auditable, and keeps data in-house.
2. **Tier 2:** documents flagged for review (unparseable output, a missing required field, a value without a
   citation, or a value that disagrees with its own evidence) go to a stronger model (hosted or larger
   self-hosted) and/or a human.
3. **Decide with data:** run the *same* evaluation harness on the hosted model before committing. The comparison
   is then accuracy, grounding, hallucination, fairness, cost and latency, all measured the same way.

## 4. How to measure the hosted side (optional, costs cents)

```bash
pip install anthropic
set ANTHROPIC_API_KEY=...            # macOS/Linux: export ANTHROPIC_API_KEY=...
python -m claimlens run --predictor anthropic --model claude-haiku-4-5 --split holdout --limit 50 --run-id hosted_haiku_holdout50
python -m claimlens cost runs/hosted_haiku_holdout50 --in-price 1 --out-price 5
python -m claimlens cost runs/sft_v2ctrl_v2_greedy_holdout --gpu-hourly 0.53
```

**Cost:** about 50 × $0.004, roughly $0.20. For a fair comparison, compare against the first 50 held-out
documents of the self-hosted run.

**Caveat:** the hosted model may resize images internally, so check its box coordinates on a few examples before
trusting the grounding numbers.

## Sources

- [Anthropic API pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [Anthropic vision docs (image token formula)](https://platform.claude.com/docs/en/build-with-claude/vision)
- [T4 cloud pricing comparison (GetDeploying)](https://getdeploying.com/gpus/nvidia-t4)
- [L4 cloud pricing comparison (GetDeploying)](https://getdeploying.com/gpus/nvidia-l4)
