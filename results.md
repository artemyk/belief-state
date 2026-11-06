### ε = 0.05

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 6.1e-5 | 0.92 ± 0.03 | 0.31 ± 0.43 | -3.02 ± 2.27 | -1.93 ± 2.01 |
| small | 8.9e-4 | 0.68 ± 0.09 | 0.25 ± 0.18 | -2.92 ± 4.08 | -6.77 ± 12.78 |
| large, RoPE | 9.5e-5 | 0.98 ± 0.00 | 0.21 ± 0.84 | 0.97 ± 0.01 | -340.44 ± 208.01 |
| small, RoPE | 1.9e-3 | 0.74 ± 0.05 | 0.71 ± 0.04 | 0.70 ± 0.05 | -148.48 ± 91.02 |

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
| large | 6.1e-5 | 0.91 ± 0.04 | 0.21 ± 0.63 | -2.60 ± 1.49 | -2.94 ± 3.48 |
| small | 8.9e-4 | 0.64 ± 0.10 | 0.22 ± 0.25 | -1.82 ± 2.14 | -12.78 ± 26.88 |
| large, RoPE | 9.5e-5 | 0.97 ± 0.01 | -0.46 ± 1.89 | 0.96 ± 0.01 | -382.30 ± 213.87 |
| small, RoPE | 1.9e-3 | 0.69 ± 0.03 | 0.66 ± 0.02 | 0.64 ± 0.03 | -203.26 ± 186.16 |

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
| odd/even | large | 0.99 | -1.30 | -23.57 |
| odd/even | small | 0.94 | -1.93 | -6.63 |
| odd/even | large, RoPE | 0.98 | -11.53 | -0.34 |
| odd/even | small, RoPE | 0.98 | -0.11 | 0.18 |
| 1st → 2nd half | large | 0.97 | -36.62 | -53.98 |
| 1st → 2nd half | small | 0.81 | -17.39 | -8.81 |
| 1st → 2nd half | large, RoPE | 1.00 | -0.07 | -0.52 |
| 1st → 2nd half | small, RoPE | 0.97 | -0.04 | 0.13 |
| 2nd → 1st half | large | 0.96 | 0.00 | -12.69 |
| 2nd → 1st half | small | 0.90 | 0.00 | -4.10 |
| 2nd → 1st half | large, RoPE | -6.67 | 0.00 | -63.16 |
| 2nd → 1st half | small, RoPE | -3.91 | 0.00 | -12.18 |

</details>

### ε = 0

**Test KL and belief-state R² by probe split**

| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |
|---|---:|---:|---:|---:|---:|
| large | 1.1e-6 | 0.87 ± 0.05 | -7.33 ± 8.33 | -94.33 ± 199.96 | -39.88 ± 79.54 |
| small | 2.0e-4 | 0.53 ± 0.08 | -0.50 ± 0.57 | -2.31 ± 3.83 | -5.83 ± 4.07 |
| large, RoPE | 4.5e-6 | 0.93 ± 0.03 | 0.74 ± 0.18 | 0.86 ± 0.10 | -154.47 ± 105.37 |
| small, RoPE | 1.6e-3 | 0.64 ± 0.04 | 0.63 ± 0.04 | 0.60 ± 0.04 | -76.83 ± 107.70 |

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
| large | 1.1e-6 | 0.87 ± 0.05 | -7.35 ± 8.35 | -94.06 ± 199.33 | -40.62 ± 81.14 |
| small | 2.0e-4 | 0.53 ± 0.08 | -0.49 ± 0.56 | -2.30 ± 3.83 | -5.83 ± 4.08 |
| large, RoPE | 4.5e-6 | 0.93 ± 0.03 | 0.74 ± 0.18 | 0.86 ± 0.10 | -154.51 ± 105.52 |
| small, RoPE | 1.6e-3 | 0.64 ± 0.04 | 0.63 ± 0.04 | 0.60 ± 0.04 | -76.70 ± 107.41 |

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
| odd/even | large | -0.73 | -67.66 | -14.36 |
| odd/even | small | 0.20 | -1.26 | -4.18 |
| odd/even | large, RoPE | 0.96 | 0.02 | -1.04 |
| odd/even | small, RoPE | 0.92 | 0.44 | 0.02 |
| 1st → 2nd half | large | -4.30 | -114.00 | -38.96 |
| 1st → 2nd half | small | -0.97 | -64.93 | -4.48 |
| 1st → 2nd half | large, RoPE | 0.99 | 0.63 | -0.90 |
| 1st → 2nd half | small, RoPE | 0.91 | 0.49 | 0.02 |
| 2nd → 1st half | large | 0.72 | -9.58 | -5.77 |
| 2nd → 1st half | small | 0.78 | -0.99 | -1.80 |
| 2nd → 1st half | large, RoPE | -2.48 | -90.03 | -280.33 |
| 2nd → 1st half | small, RoPE | -2.21 | -37.76 | -28.70 |

</details>
