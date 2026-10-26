### ε = 0.05

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 6.1e-5 | 0.93 ± 0.03 | 0.83 ± 0.04 | 0.71 ± 0.12 | 0.51 ± 0.33 |
| small | 8.9e-4 | 0.69 ± 0.10 | 0.60 ± 0.10 | 0.48 ± 0.08 | 0.48 ± 0.04 |
| large, RoPE | 9.5e-5 | 0.98 ± 0.00 | 0.95 ± 0.02 | 0.90 ± 0.05 | -546.25 ± 287.63 |
| small, RoPE | 1.9e-3 | 0.74 ± 0.04 | 0.64 ± 0.09 | 0.61 ± 0.05 | -192.55 ± 121.97 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 1.00 | 0.13 | 0.24 |
| small | 0.97 | 0.07 | 0.20 |
| large, RoPE | 1.00 | 0.15 | 0.32 |
| small, RoPE | 0.98 | 0.07 | 0.26 |

<details>
<summary>Process weighting</summary>

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 6.1e-5 | 0.92 ± 0.04 | 0.81 ± 0.06 | 0.67 ± 0.15 | 0.41 ± 0.52 |
| small | 8.9e-4 | 0.65 ± 0.11 | 0.55 ± 0.11 | 0.40 ± 0.08 | 0.47 ± 0.05 |
| large, RoPE | 9.5e-5 | 0.97 ± 0.01 | 0.94 ± 0.03 | 0.87 ± 0.08 | -590.84 ± 327.79 |
| small, RoPE | 1.9e-3 | 0.70 ± 0.03 | 0.61 ± 0.06 | 0.53 ± 0.04 | -265.36 ± 229.37 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 1.00 | 0.16 | 0.19 |
| small | 0.98 | 0.11 | 0.16 |
| large, RoPE | 1.00 | 0.17 | 0.27 |
| small, RoPE | 0.97 | 0.11 | 0.21 |

</details>

<details>
<summary>R² of the controls, held-out splits</summary>

| split | model | next-token distribution | shuffled belief state | untrained network |
|---|---|---:|---:|---:|
| odd/even | large | 1.00 | -0.02 | 0.16 |
| odd/even | small | 0.97 | -0.01 | 0.16 |
| odd/even | large, RoPE | 1.00 | -0.01 | -0.22 |
| odd/even | small, RoPE | 0.97 | 0.00 | -1.98 |
| 1st → 2nd half | large | 0.99 | -0.01 | 0.15 |
| 1st → 2nd half | small | 0.96 | -0.01 | 0.15 |
| 1st → 2nd half | large, RoPE | 1.00 | -0.01 | 0.16 |
| 1st → 2nd half | small, RoPE | 0.97 | 0.00 | 0.16 |
| 2nd → 1st half | large | 0.99 | 0.00 | 0.18 |
| 2nd → 1st half | small | 0.98 | 0.00 | 0.17 |
| 2nd → 1st half | large, RoPE | -25.10 | 0.00 | -158.94 |
| 2nd → 1st half | small, RoPE | -6.10 | 0.00 | -43.35 |

</details>

### ε = 0

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 1.1e-6 | 0.88 ± 0.05 | 0.69 ± 0.09 | 0.56 ± 0.13 | 0.17 ± 0.20 |
| small | 2.0e-4 | 0.54 ± 0.09 | 0.45 ± 0.11 | 0.35 ± 0.06 | -0.36 ± 1.49 |
| large, RoPE | 4.5e-6 | 0.93 ± 0.03 | 0.85 ± 0.04 | 0.73 ± 0.09 | -230.55 ± 232.43 |
| small, RoPE | 1.6e-3 | 0.65 ± 0.03 | 0.60 ± 0.04 | 0.47 ± 0.06 | -103.07 ± 143.37 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 1.00 | 0.83 | 0.12 |
| small | 0.94 | 0.48 | 0.10 |
| large, RoPE | 1.00 | 0.83 | 0.19 |
| small, RoPE | 0.92 | 0.52 | 0.14 |

<details>
<summary>R² of the controls, held-out splits</summary>

| split | model | next-token distribution | shuffled belief state | untrained network |
|---|---|---:|---:|---:|
| odd/even | large | 0.98 | 0.44 | 0.08 |
| odd/even | small | 0.92 | 0.29 | 0.08 |
| odd/even | large, RoPE | 0.99 | 0.51 | 0.08 |
| odd/even | small, RoPE | 0.91 | 0.20 | -1.19 |
| 1st → 2nd half | large | 0.98 | 0.54 | 0.07 |
| 1st → 2nd half | small | 0.95 | 0.37 | 0.07 |
| 1st → 2nd half | large, RoPE | 0.99 | 0.59 | 0.07 |
| 1st → 2nd half | small, RoPE | 0.92 | 0.41 | 0.07 |
| 2nd → 1st half | large | 0.92 | 0.13 | 0.11 |
| 2nd → 1st half | small | 0.83 | -0.10 | 0.10 |
| 2nd → 1st half | large, RoPE | -5.83 | -162.56 | -126.47 |
| 2nd → 1st half | small, RoPE | -3.52 | -42.15 | -31.79 |

</details>
