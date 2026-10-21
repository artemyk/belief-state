### 4 layers, d_model 64

| ε | belief state | next-token distribution | shuffled belief state | untrained network |
|---:|---:|---:|---:|---:|
| 0.05 | **0.93 ± 0.01** | 1.00 ± 0.00 | 0.11 ± 0.01 | 0.24 ± 0.01 |
| 0.01 | **0.91 ± 0.01** | 0.98 ± 0.00 | 0.15 ± 0.01 | 0.19 ± 0.01 |
| 0 | **0.81 ± 0.03** | 1.00 ± 0.00 | 0.84 ± 0.01 | 0.12 ± 0.01 |

<details>
<summary>Process weighting</summary>

| ε | belief state | next-token distribution | shuffled belief state | untrained network |
|---:|---:|---:|---:|---:|
| 0.05 | **0.92 ± 0.01** | 1.00 ± 0.00 | 0.12 ± 0.01 | 0.20 ± 0.01 |
| 0.01 | **0.93 ± 0.01** | 1.00 ± 0.00 | 0.13 ± 0.01 | 0.13 ± 0.01 |
| 0 | **0.81 ± 0.03** | 1.00 ± 0.00 | 0.84 ± 0.01 | 0.12 ± 0.01 |

</details>

### 2 layers, d_model 32

| ε | belief state | next-token distribution | shuffled belief state | untrained network |
|---:|---:|---:|---:|---:|
| 0.05 | **0.73 ± 0.03** | 0.98 ± 0.00 | 0.06 ± 0.01 | 0.20 ± 0.00 |
| 0.01 | **0.70 ± 0.03** | 0.94 ± 0.01 | 0.08 ± 0.01 | 0.15 ± 0.00 |
| 0 | **0.60 ± 0.02** | 0.96 ± 0.00 | 0.65 ± 0.01 | 0.10 ± 0.00 |

<details>
<summary>Process weighting</summary>

| ε | belief state | next-token distribution | shuffled belief state | untrained network |
|---:|---:|---:|---:|---:|
| 0.05 | **0.69 ± 0.04** | 0.99 ± 0.00 | 0.07 ± 0.01 | 0.16 ± 0.00 |
| 0.01 | **0.69 ± 0.03** | 0.98 ± 0.00 | 0.07 ± 0.00 | 0.11 ± 0.00 |
| 0 | **0.60 ± 0.02** | 0.96 ± 0.00 | 0.65 ± 0.01 | 0.10 ± 0.00 |

</details>

Did not reach the training-KL target: ε = 0.05 (5 of 5 seeds); ε = 0.01 (5 of 5 seeds).
