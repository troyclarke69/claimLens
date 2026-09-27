| Metric | `baseline_v2_greedy_holdout` | `sft_v1_v2_greedy_holdout` |
|---|---|---|
| Valid JSON | 99.0% | 99.0% |
| Field accuracy | 90.1% | 89.9% (▼0.3 pts ✗) |
| Grounded accuracy | 38.2% | 84.4% (▲46.2 pts ✓) |
| Evidence supported | 45.2% | 89.3% (▲44.1 pts ✓) |
| Hallucination rate | 8.7% | 4.9% (▼3.9 pts ✓) |
| Miss rate | 3.7% | 5.2% (▲1.5 pts ✗) |
| Document exact match | 53.0% | 49.0% (▼4.0 pts ✗) |
| Mean reward | 0.774 | 0.892 (▲0.118 ✓) |

**Per-field grounded accuracy**

| Field | `baseline_v2_greedy_holdout` | `sft_v1_v2_greedy_holdout` |
|---|---|---|
| `claim_number` | 10.8% | 75.3% |
| `policy_number` | 25.8% | 87.9% |
| `claimant_name` | 66.0% | 95.0% |
| `date_of_loss` | 12.0% | 59.0% |
| `provider_name` | 63.6% | 97.7% |
| `incident_type` | 44.0% | 86.0% |
| `total_amount` | 45.0% | 93.0% |

**Error types**

| Error type | `baseline_v2_greedy_holdout` | `sft_v1_v2_greedy_holdout` |
|---|---|---|
| diacritics | 4 | 1 |
| distractor_confusion | 9 | 1 |
| distractor_for_absent | 6 | 2 |
| field_swap | 2 | 1 |
| hallucinated | 3 | 3 |
| missed | 15 | 24 |
| near_miss | 20 | 26 |
| unparseable | 7 | 7 |
| wrong_value | 3 | 6 |
