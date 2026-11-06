### ε = 0.05

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 6.1e-5 | 0.92 ± 0.03 | 0.85 ± 0.05 | 0.76 ± 0.05 | 0.60 ± 0.23 |
| small | 8.9e-4 | 0.68 ± 0.09 | 0.60 ± 0.10 | 0.49 ± 0.08 | 0.52 ± 0.06 |
| large, RoPE | 9.5e-5 | 0.98 ± 0.00 | 0.97 ± 0.01 | 0.93 ± 0.02 | -340.44 ± 208.00 |
| small, RoPE | 1.9e-3 | 0.74 ± 0.05 | 0.70 ± 0.03 | 0.64 ± 0.02 | -148.48 ± 91.02 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 1.00 | 0.11 | 0.24 |
| small | 0.97 | 0.06 | 0.20 |
| large, RoPE | 1.00 | 0.13 | 0.32 |
| small, RoPE | 0.98 | 0.06 | 0.26 |

<details>
<summary>Process weighting</summary>

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 6.1e-5 | 0.91 ± 0.04 | 0.83 ± 0.07 | 0.74 ± 0.08 | 0.53 ± 0.35 |
| small | 8.9e-4 | 0.64 ± 0.10 | 0.56 ± 0.11 | 0.41 ± 0.08 | 0.50 ± 0.06 |
| large, RoPE | 9.5e-5 | 0.97 ± 0.01 | 0.95 ± 0.02 | 0.91 ± 0.05 | -382.30 ± 213.87 |
| small, RoPE | 1.9e-3 | 0.69 ± 0.03 | 0.65 ± 0.03 | 0.56 ± 0.03 | -203.26 ± 186.16 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 1.00 | 0.13 | 0.19 |
| small | 0.98 | 0.06 | 0.16 |
| large, RoPE | 1.00 | 0.15 | 0.27 |
| small, RoPE | 0.97 | 0.07 | 0.21 |

</details>

<details>
<summary>R² of the controls, held-out splits</summary>

| split | model | next-token distribution | shuffled belief state | untrained network |
|---|---|---:|---:|---:|
| odd/even | large | 1.00 | -0.01 | 0.16 |
| odd/even | small | 0.97 | -0.01 | 0.16 |
| odd/even | large, RoPE | 1.00 | -0.02 | 0.22 |
| odd/even | small, RoPE | 0.97 | -0.01 | 0.23 |
| 1st → 2nd half | large | 0.99 | -0.01 | 0.15 |
| 1st → 2nd half | small | 0.96 | 0.00 | 0.15 |
| 1st → 2nd half | large, RoPE | 1.00 | -0.01 | 0.16 |
| 1st → 2nd half | small, RoPE | 0.97 | -0.01 | 0.17 |
| 2nd → 1st half | large | 0.99 | 0.00 | 0.18 |
| 2nd → 1st half | small | 0.98 | 0.00 | 0.18 |
| 2nd → 1st half | large, RoPE | -6.67 | 0.00 | -27.61 |
| 2nd → 1st half | small, RoPE | -3.91 | 0.00 | -7.11 |

</details>

### ε = 0

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 1.1e-6 | 0.87 ± 0.05 | 0.71 ± 0.07 | 0.61 ± 0.10 | 0.40 ± 0.14 |
| small | 2.0e-4 | 0.53 ± 0.08 | 0.45 ± 0.11 | 0.36 ± 0.07 | -0.16 ± 1.13 |
| large, RoPE | 4.5e-6 | 0.93 ± 0.03 | 0.89 ± 0.04 | 0.74 ± 0.08 | -135.82 ± 129.16 |
| small, RoPE | 1.6e-3 | 0.64 ± 0.04 | 0.62 ± 0.03 | 0.48 ± 0.06 | -76.83 ± 107.70 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 1.00 | 0.86 | 0.12 |
| small | 0.94 | 0.59 | 0.10 |
| large, RoPE | 1.00 | 0.88 | 0.19 |
| small, RoPE | 0.92 | 0.59 | 0.14 |

<details>
<summary>Process weighting</summary>

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 1.1e-6 | 0.87 ± 0.05 | 0.71 ± 0.07 | 0.61 ± 0.10 | 0.40 ± 0.14 |
| small | 2.0e-4 | 0.53 ± 0.08 | 0.45 ± 0.11 | 0.36 ± 0.07 | -0.16 ± 1.13 |
| large, RoPE | 4.5e-6 | 0.93 ± 0.03 | 0.89 ± 0.04 | 0.74 ± 0.08 | -135.86 ± 129.30 |
| small, RoPE | 1.6e-3 | 0.64 ± 0.04 | 0.62 ± 0.03 | 0.48 ± 0.06 | -76.70 ± 107.41 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 1.00 | 0.86 | 0.12 |
| small | 0.94 | 0.59 | 0.10 |
| large, RoPE | 1.00 | 0.88 | 0.19 |
| small, RoPE | 0.92 | 0.59 | 0.14 |

</details>

<details>
<summary>R² of the controls, held-out splits</summary>

| split | model | next-token distribution | shuffled belief state | untrained network |
|---|---|---:|---:|---:|
| odd/even | large | 0.98 | 0.58 | 0.08 |
| odd/even | small | 0.92 | 0.40 | 0.08 |
| odd/even | large, RoPE | 0.99 | 0.65 | 0.11 |
| odd/even | small, RoPE | 0.91 | 0.46 | 0.12 |
| 1st → 2nd half | large | 0.98 | 0.57 | 0.07 |
| 1st → 2nd half | small | 0.94 | 0.37 | 0.07 |
| 1st → 2nd half | large, RoPE | 0.99 | 0.67 | 0.07 |
| 1st → 2nd half | small, RoPE | 0.92 | 0.47 | 0.07 |
| 2nd → 1st half | large | 0.93 | 0.25 | 0.11 |
| 2nd → 1st half | small | 0.83 | -0.15 | 0.11 |
| 2nd → 1st half | large, RoPE | -2.48 | -90.03 | -11.97 |
| 2nd → 1st half | small, RoPE | -2.20 | -37.75 | -2.98 |

</details>
