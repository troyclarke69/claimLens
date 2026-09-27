| Metric | `baseline_v2_greedy_fairness` | `sft_v1_v2_greedy_fairness` |
|---|---|---|
| Valid JSON | 100.0% | 100.0% |
| Field accuracy | 93.4% | 97.4% (▲4.0 pts ✓) |
| Grounded accuracy | 32.9% | 96.8% (▲63.9 pts ✓) |
| Evidence supported | 46.1% | 96.8% (▲50.8 pts ✓) |
| Hallucination rate | 12.1% | 0.0% (▼12.1 pts ✓) |
| Miss rate | 0.0% | 0.0% |
| Document exact match | 58.0% | 84.0% (▲26.0 pts ✓) |
| Mean reward | 0.802 | 0.975 (▲0.173 ✓) |
| Counterfactual consistency | 92.0% | 100.0% |

**Per-field grounded accuracy**

| Field | `baseline_v2_greedy_fairness` | `sft_v1_v2_greedy_fairness` |
|---|---|---|
| `claim_number` | 19.1% | 90.5% |
| `policy_number` | 14.3% | 100.0% |
| `claimant_name` | 51.0% | 94.0% |
| `date_of_loss` | 19.0% | 100.0% |
| `provider_name` | 68.4% | 100.0% |
| `incident_type` | 11.5% | 100.0% |
| `total_amount` | 35.0% | 96.0% |

**Error types**

| Error type | `baseline_v2_greedy_fairness` | `sft_v1_v2_greedy_fairness` |
|---|---|---|
| diacritics | 1 | 0 |
| distractor_confusion | 4 | 0 |
| distractor_for_absent | 16 | 0 |
| near_miss | 22 | 17 |
| wrong_value | 3 | 1 |
