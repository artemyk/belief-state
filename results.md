### ε = 0.05

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 6.1e-5 | 0.90 ± 0.04 | 0.85 ± 0.05 | 0.71 ± 0.09 | 0.30 ± 0.38 |
| small | 8.9e-4 | 0.64 ± 0.10 | 0.60 ± 0.10 | 0.45 ± 0.10 | 0.47 ± 0.03 |
| large, RoPE | 9.5e-5 | 0.97 ± 0.01 | 0.97 ± 0.01 | 0.96 ± 0.01 | -340.44 ± 208.01 |
| small, RoPE | 1.9e-3 | 0.72 ± 0.04 | 0.70 ± 0.04 | 0.69 ± 0.04 | -148.48 ± 91.02 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 1.00 | 0.02 | 0.18 |
| small | 0.97 | 0.01 | 0.17 |
| large, RoPE | 1.00 | 0.01 | 0.26 |
| small, RoPE | 0.97 | 0.01 | 0.24 |

<details>
<summary>Process weighting</summary>

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 6.1e-5 | 0.89 ± 0.05 | 0.84 ± 0.07 | 0.69 ± 0.14 | 0.20 ± 0.50 |
| small | 8.9e-4 | 0.60 ± 0.11 | 0.56 ± 0.11 | 0.41 ± 0.09 | 0.45 ± 0.04 |
| large, RoPE | 9.5e-5 | 0.97 ± 0.01 | 0.95 ± 0.02 | 0.95 ± 0.02 | -382.30 ± 213.87 |
| small, RoPE | 1.9e-3 | 0.67 ± 0.03 | 0.66 ± 0.02 | 0.63 ± 0.03 | -203.26 ± 186.16 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 1.00 | 0.02 | 0.15 |
| small | 0.98 | 0.01 | 0.15 |
| large, RoPE | 1.00 | 0.02 | 0.21 |
| small, RoPE | 0.97 | 0.01 | 0.20 |

</details>

<details>
<summary>R² of the controls, held-out splits</summary>

| split | model | next-token distribution | shuffled belief state | untrained network |
|---|---|---:|---:|---:|
| odd/even | large | 1.00 | -0.01 | 0.16 |
| odd/even | small | 0.97 | -0.01 | 0.16 |
| odd/even | large, RoPE | 1.00 | -0.02 | 0.23 |
| odd/even | small, RoPE | 0.97 | -0.01 | 0.23 |
| 1st → 2nd half | large | 0.99 | -0.01 | 0.15 |
| 1st → 2nd half | small | 0.96 | 0.00 | 0.15 |
| 1st → 2nd half | large, RoPE | 1.00 | -0.01 | 0.15 |
| 1st → 2nd half | small, RoPE | 0.97 | 0.00 | 0.17 |
| 2nd → 1st half | large | 0.99 | 0.00 | 0.18 |
| 2nd → 1st half | small | 0.98 | 0.00 | 0.18 |
| 2nd → 1st half | large, RoPE | -6.67 | 0.00 | -40.41 |
| 2nd → 1st half | small, RoPE | -3.91 | 0.00 | -11.70 |

</details>

### ε = 0

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 1.1e-6 | 0.79 ± 0.09 | 0.71 ± 0.07 | 0.68 ± 0.07 | 0.34 ± 0.16 |
| small | 2.0e-4 | 0.49 ± 0.10 | 0.45 ± 0.11 | 0.40 ± 0.10 | -1.81 ± 4.82 |
| large, RoPE | 4.5e-6 | 0.92 ± 0.03 | 0.89 ± 0.04 | 0.87 ± 0.07 | -154.47 ± 105.37 |
| small, RoPE | 1.6e-3 | 0.64 ± 0.03 | 0.62 ± 0.03 | 0.59 ± 0.03 | -76.83 ± 107.70 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 0.99 | 0.74 | 0.09 |
| small | 0.93 | 0.52 | 0.09 |
| large, RoPE | 1.00 | 0.78 | 0.13 |
| small, RoPE | 0.92 | 0.56 | 0.13 |

<details>
<summary>Process weighting</summary>

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 1.1e-6 | 0.79 ± 0.09 | 0.71 ± 0.07 | 0.68 ± 0.07 | 0.34 ± 0.15 |
| small | 2.0e-4 | 0.49 ± 0.10 | 0.45 ± 0.11 | 0.40 ± 0.10 | -1.82 ± 4.84 |
| large, RoPE | 4.5e-6 | 0.92 ± 0.03 | 0.89 ± 0.04 | 0.87 ± 0.07 | -154.51 ± 105.52 |
| small, RoPE | 1.6e-3 | 0.64 ± 0.03 | 0.62 ± 0.03 | 0.59 ± 0.03 | -76.70 ± 107.41 |

**R² of the controls, random split**

| model | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|
| large | 0.99 | 0.74 | 0.09 |
| small | 0.93 | 0.52 | 0.09 |
| large, RoPE | 1.00 | 0.77 | 0.13 |
| small, RoPE | 0.92 | 0.56 | 0.13 |

</details>

<details>
<summary>R² of the controls, held-out splits</summary>

| split | model | next-token distribution | shuffled belief state | untrained network |
|---|---|---:|---:|---:|
| odd/even | large | 0.98 | 0.59 | 0.08 |
| odd/even | small | 0.92 | 0.40 | 0.08 |
| odd/even | large, RoPE | 0.99 | 0.65 | 0.11 |
| odd/even | small, RoPE | 0.91 | 0.46 | 0.12 |
| 1st → 2nd half | large | 0.99 | 0.53 | 0.07 |
| 1st → 2nd half | small | 0.94 | 0.31 | 0.07 |
| 1st → 2nd half | large, RoPE | 0.99 | 0.67 | 0.06 |
| 1st → 2nd half | small, RoPE | 0.91 | 0.41 | 0.08 |
| 2nd → 1st half | large | 0.93 | 0.27 | 0.11 |
| 2nd → 1st half | small | 0.83 | -0.19 | 0.11 |
| 2nd → 1st half | large, RoPE | -2.48 | -90.03 | -19.51 |
| 2nd → 1st half | small, RoPE | -2.21 | -37.75 | -5.01 |

</details>
