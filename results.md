### ε = 0.05

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 4.8e-5 | 0.93 ± 0.02 | 0.84 ± 0.05 | 0.72 ± 0.08 | 0.60 ± 0.18 |
| small | 9.9e-4 | 0.69 ± 0.10 | 0.60 ± 0.11 | 0.51 ± 0.07 | 0.46 ± 0.08 |
| large, RoPE | 9.2e-4 | 0.96 ± 0.00 | 0.95 ± 0.01 | 0.90 ± 0.02 | 0.89 ± 0.03 |
| small, RoPE | 3.1e-3 | 0.75 ± 0.04 | 0.73 ± 0.04 | 0.65 ± 0.05 | 0.70 ± 0.05 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 1.00 | 0.11 | 0.25 |
| small | 0.97 | 0.05 | 0.20 |
| large, RoPE | 0.98 | 0.08 | 0.30 |
| small, RoPE | 0.96 | 0.03 | 0.25 |

<details>
<summary>Process weighting</summary>

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 4.8e-5 | 0.92 ± 0.03 | 0.83 ± 0.06 | 0.68 ± 0.09 | 0.30 ± 0.42 |
| small | 9.9e-4 | 0.65 ± 0.11 | 0.56 ± 0.10 | 0.46 ± 0.07 | 0.44 ± 0.06 |
| large, RoPE | 9.2e-4 | 0.96 ± 0.00 | 0.94 ± 0.01 | 0.86 ± 0.02 | 0.86 ± 0.04 |
| small, RoPE | 3.1e-3 | 0.69 ± 0.04 | 0.67 ± 0.04 | 0.55 ± 0.03 | 0.65 ± 0.04 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 1.00 | 0.13 | 0.20 |
| small | 0.98 | 0.06 | 0.17 |
| large, RoPE | 0.99 | 0.10 | 0.26 |
| small, RoPE | 0.96 | 0.05 | 0.21 |

</details>

<details>
<summary>R² of the controls, held-out splits</summary>

| split | model | next-token distribution | shuffled belief state | untrained network |
|---|---|---:|---:|---:|
| odd/even | large | 1.00 | -0.01 | 0.16 |
| odd/even | small | 0.97 | -0.01 | 0.16 |
| odd/even | large, RoPE | 0.97 | -0.02 | 0.25 |
| odd/even | small, RoPE | 0.95 | -0.01 | 0.23 |
| 1st → 2nd half | large | 0.99 | -0.01 | 0.15 |
| 1st → 2nd half | small | 0.96 | -0.01 | 0.15 |
| 1st → 2nd half | large, RoPE | 0.97 | -0.01 | 0.16 |
| 1st → 2nd half | small, RoPE | 0.97 | -0.01 | 0.16 |
| 2nd → 1st half | large | 0.98 | 0.00 | 0.18 |
| 2nd → 1st half | small | 0.97 | 0.00 | 0.18 |
| 2nd → 1st half | large, RoPE | 0.96 | 0.00 | -0.24 |
| 2nd → 1st half | small, RoPE | 0.95 | 0.00 | -3.06 |

</details>

### ε = 0

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 1.1e-6 | 0.83 ± 0.07 | 0.67 ± 0.05 | 0.56 ± 0.06 | 0.30 ± 0.22 |
| small | 4.8e-5 | 0.56 ± 0.04 | 0.49 ± 0.06 | 0.41 ± 0.06 | 0.31 ± 0.16 |
| large, RoPE | 1.2e-3 | 0.92 ± 0.01 | 0.88 ± 0.03 | 0.75 ± 0.09 | -2.76 ± 7.48 |
| small, RoPE | 5.9e-3 | 0.61 ± 0.04 | 0.59 ± 0.03 | 0.43 ± 0.03 | -0.86 ± 2.10 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 1.00 | 0.86 | 0.12 |
| small | 0.95 | 0.62 | 0.10 |
| large, RoPE | 0.99 | 0.83 | 0.18 |
| small, RoPE | 0.89 | 0.48 | 0.14 |

<details>
<summary>R² of the controls, held-out splits</summary>

| split | model | next-token distribution | shuffled belief state | untrained network |
|---|---|---:|---:|---:|
| odd/even | large | 0.99 | 0.60 | 0.09 |
| odd/even | small | 0.93 | 0.43 | 0.08 |
| odd/even | large, RoPE | 0.97 | 0.66 | 0.12 |
| odd/even | small, RoPE | 0.87 | 0.37 | 0.11 |
| 1st → 2nd half | large | 0.99 | 0.57 | 0.07 |
| 1st → 2nd half | small | 0.96 | 0.41 | 0.07 |
| 1st → 2nd half | large, RoPE | 0.97 | 0.72 | 0.07 |
| 1st → 2nd half | small, RoPE | 0.88 | 0.24 | 0.07 |
| 2nd → 1st half | large | 0.96 | 0.28 | 0.11 |
| 2nd → 1st half | small | 0.84 | 0.04 | 0.11 |
| 2nd → 1st half | large, RoPE | 0.86 | -0.28 | -14.78 |
| 2nd → 1st half | small, RoPE | 0.70 | -1.08 | -0.86 |

</details>
