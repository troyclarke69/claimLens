| Metric | `sft_v1_v2_greedy_holdout` | `sft_v2ctrl_v2_greedy_holdout` | `grpo_v1_v2_greedy_holdout` |
|---|---|---|---|
| Valid JSON | 99.0% | 99.0% | 99.0% |
| Field accuracy | 89.9% | 92.9% (▲3.0 pts ✓) | 89.9% |
| Grounded accuracy | 84.4% | 86.4% (▲2.0 pts ✓) | 82.9% (▼1.5 pts ✗) |
| Evidence supported | 89.3% | 91.0% (▲1.7 pts ✓) | 88.9% (▼0.4 pts ✗) |
| Hallucination rate | 4.9% | 2.9% (▼1.9 pts ✓) | 1.9% (▼2.9 pts ✓) |
| Miss rate | 5.2% | 3.9% (▼1.3 pts ✓) | 6.7% (▲1.5 pts ✗) |
| Document exact match | 49.0% | 63.0% (▲14.0 pts ✓) | 45.0% (▼4.0 pts ✗) |
| Mean reward | 0.892 | 0.917 (▲0.025 ✓) | 0.888 (▼0.004 ✗) |

**Per-field grounded accuracy**

| Field | `sft_v1_v2_greedy_holdout` | `sft_v2ctrl_v2_greedy_holdout` | `grpo_v1_v2_greedy_holdout` |
|---|---|---|---|
| `claim_number` | 75.3% | 77.4% | 67.7% |
| `policy_number` | 87.9% | 89.4% | 87.9% |
| `claimant_name` | 95.0% | 95.0% | 95.0% |
| `date_of_loss` | 59.0% | 64.0% | 58.0% |
| `provider_name` | 97.7% | 97.7% | 95.5% |
| `incident_type` | 86.0% | 92.0% | 88.0% |
| `total_amount` | 93.0% | 94.0% | 93.0% |

**Error types**

| Error type | `sft_v1_v2_greedy_holdout` | `sft_v2ctrl_v2_greedy_holdout` | `grpo_v1_v2_greedy_holdout` |
|---|---|---|---|
| diacritics | 1 | 1 | 1 |
| distractor_confusion | 1 | 1 | 0 |
| distractor_for_absent | 2 | 1 | 1 |
| field_swap | 1 | 1 | 0 |
| hallucinated | 3 | 2 | 1 |
| missed | 24 | 16 | 33 |
| near_miss | 26 | 14 | 24 |
| unparseable | 7 | 7 | 7 |
| wrong_value | 6 | 7 | 4 |
