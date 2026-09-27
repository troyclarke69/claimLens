| Metric | `baseline_v2_greedy_test` | `sft_v1_v2_greedy_test` |
|---|---|---|
| Valid JSON | 100.0% | 100.0% |
| Field accuracy | 93.0% | 97.0% (▲4.0 pts ✓) |
| Grounded accuracy | 37.6% | 96.1% (▲58.5 pts ✓) |
| Evidence supported | 47.8% | 96.4% (▲48.6 pts ✓) |
| Hallucination rate | 7.1% | 0.0% (▼7.1 pts ✓) |
| Miss rate | 0.3% | 0.0% (▼0.3 pts ✓) |
| Document exact match | 64.0% | 84.0% (▲20.0 pts ✓) |
| Mean reward | 0.805 | 0.970 (▲0.165 ✓) |

**Per-field grounded accuracy**

| Field | `baseline_v2_greedy_test` | `sft_v1_v2_greedy_test` |
|---|---|---|
| `claim_number` | 18.7% | 89.0% |
| `policy_number` | 32.8% | 98.4% |
| `claimant_name` | 54.0% | 97.0% |
| `date_of_loss` | 28.0% | 96.0% |
| `provider_name` | 63.4% | 100.0% |
| `incident_type` | 25.5% | 96.1% |
| `total_amount` | 36.0% | 97.0% |

**Error types**

| Error type | `baseline_v2_greedy_test` | `sft_v1_v2_greedy_test` |
|---|---|---|
| diacritics | 1 | 0 |
| distractor_confusion | 4 | 0 |
| distractor_for_absent | 6 | 0 |
| field_swap | 2 | 2 |
| hallucinated | 2 | 0 |
| missed | 2 | 0 |
| near_miss | 30 | 19 |
| wrong_value | 2 | 0 |
