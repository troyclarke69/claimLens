| Metric | `sft_v1_v2_greedy_test` | `sft_v2ctrl_v2_greedy_test` | `grpo_v1_v2_greedy_test` |
|---|---|---|---|
| Valid JSON | 100.0% | 100.0% | 100.0% |
| Field accuracy | 97.0% | 97.4% (▲0.4 pts ✓) | 97.1% (▲0.1 pts ✓) |
| Grounded accuracy | 96.1% | 96.1% | 95.9% (▼0.2 pts ✗) |
| Evidence supported | 96.4% | 96.6% (▲0.2 pts ✓) | 96.3% (▼0.2 pts ✗) |
| Hallucination rate | 0.0% | 0.0% | 0.0% |
| Miss rate | 0.0% | 0.0% | 0.0% |
| Document exact match | 84.0% | 87.0% (▲3.0 pts ✓) | 85.0% (▲1.0 pts ✓) |
| Mean reward | 0.970 | 0.974 (▲0.004 ✓) | 0.970 (▲0.000 ✓) |

**Per-field grounded accuracy**

| Field | `sft_v1_v2_greedy_test` | `sft_v2ctrl_v2_greedy_test` | `grpo_v1_v2_greedy_test` |
|---|---|---|---|
| `claim_number` | 89.0% | 90.1% | 90.1% |
| `policy_number` | 98.4% | 98.4% | 98.4% |
| `claimant_name` | 97.0% | 96.0% | 97.0% |
| `date_of_loss` | 96.0% | 96.0% | 95.0% |
| `provider_name` | 100.0% | 100.0% | 100.0% |
| `incident_type` | 96.1% | 98.0% | 96.1% |
| `total_amount` | 97.0% | 96.0% | 96.0% |

**Error types**

| Error type | `sft_v1_v2_greedy_test` | `sft_v2ctrl_v2_greedy_test` | `grpo_v1_v2_greedy_test` |
|---|---|---|---|
| field_swap | 2 | 1 | 1 |
| near_miss | 19 | 17 | 19 |
