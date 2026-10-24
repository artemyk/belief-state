### Random split

| model | ε | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|
| large | 0.05 | **0.93 ± 0.01** | 1.00 ± 0.00 | 0.11 ± 0.01 | 0.24 ± 0.01 |
| large | 0 | **0.81 ± 0.03** | 1.00 ± 0.00 | 0.84 ± 0.01 | 0.12 ± 0.01 |
| large, RoPE | 0.05 | **0.97 ± 0.00** | 0.98 ± 0.00 | 0.08 ± 0.01 | 0.30 ± 0.01 |
| small | 0.05 | **0.73 ± 0.03** | 0.98 ± 0.00 | 0.06 ± 0.01 | 0.20 ± 0.00 |
| small | 0 | **0.60 ± 0.02** | 0.96 ± 0.00 | 0.65 ± 0.01 | 0.10 ± 0.00 |

<details>
<summary>Process weighting</summary>

| model | ε | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|
| large | 0.05 | **0.92 ± 0.01** | 1.00 ± 0.00 | 0.12 ± 0.01 | 0.20 ± 0.01 |
| large | 0 | **0.81 ± 0.03** | 1.00 ± 0.00 | 0.84 ± 0.01 | 0.12 ± 0.01 |
| large, RoPE | 0.05 | **0.96 ± 0.01** | 0.99 ± 0.00 | 0.10 ± 0.01 | 0.26 ± 0.01 |
| small | 0.05 | **0.69 ± 0.04** | 0.99 ± 0.00 | 0.07 ± 0.01 | 0.16 ± 0.00 |
| small | 0 | **0.60 ± 0.02** | 0.96 ± 0.00 | 0.65 ± 0.01 | 0.10 ± 0.00 |

</details>

### Odd/even split

| model | ε | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|
| large | 0.05 | **0.85 ± 0.03** | 0.99 ± 0.00 | -0.01 ± 0.00 | 0.16 ± 0.00 |
| large | 0 | **0.67 ± 0.01** | 0.99 ± 0.01 | 0.59 ± 0.01 | 0.08 ± 0.00 |
| large, RoPE | 0.05 | **0.95 ± 0.01** | 0.97 ± 0.00 | -0.01 ± 0.00 | 0.24 ± 0.02 |
| small | 0.05 | **0.63 ± 0.04** | 0.98 ± 0.00 | 0.00 ± 0.00 | 0.16 ± 0.00 |
| small | 0 | **0.56 ± 0.05** | 0.95 ± 0.00 | 0.49 ± 0.02 | 0.08 ± 0.00 |

<details>
<summary>Process weighting</summary>

| model | ε | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|
| large | 0.05 | **0.83 ± 0.03** | 1.00 ± 0.00 | -0.01 ± 0.00 | 0.14 ± 0.00 |
| large | 0 | **0.67 ± 0.01** | 0.99 ± 0.01 | 0.58 ± 0.01 | 0.08 ± 0.00 |
| large, RoPE | 0.05 | **0.95 ± 0.01** | 0.98 ± 0.00 | 0.00 ± 0.00 | 0.20 ± 0.01 |
| small | 0.05 | **0.58 ± 0.04** | 0.99 ± 0.00 | 0.00 ± 0.00 | 0.14 ± 0.00 |
| small | 0 | **0.56 ± 0.05** | 0.95 ± 0.01 | 0.49 ± 0.02 | 0.08 ± 0.00 |

</details>

### First half to second half

| model | ε | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|
| large | 0.05 | **0.75 ± 0.06** | 0.99 ± 0.00 | -0.01 ± 0.00 | 0.15 ± 0.01 |
| large | 0 | **0.60 ± 0.03** | 0.99 ± 0.00 | 0.47 ± 0.13 | 0.07 ± 0.00 |
| large, RoPE | 0.05 | **0.96 ± 0.01** | 0.99 ± 0.00 | -0.01 ± 0.00 | 0.17 ± 0.01 |
| small | 0.05 | **0.54 ± 0.01** | 0.97 ± 0.00 | -0.01 ± 0.00 | 0.14 ± 0.01 |
| small | 0 | **0.49 ± 0.06** | 0.97 ± 0.00 | 0.52 ± 0.03 | 0.07 ± 0.00 |

<details>
<summary>Process weighting</summary>

| model | ε | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|
| large | 0.05 | **0.70 ± 0.06** | 0.99 ± 0.00 | -0.01 ± 0.00 | 0.12 ± 0.00 |
| large | 0 | **0.60 ± 0.03** | 0.99 ± 0.00 | 0.47 ± 0.13 | 0.07 ± 0.00 |
| large, RoPE | 0.05 | **0.94 ± 0.01** | 0.99 ± 0.00 | -0.01 ± 0.00 | 0.13 ± 0.00 |
| small | 0.05 | **0.48 ± 0.02** | 0.98 ± 0.00 | -0.01 ± 0.00 | 0.12 ± 0.01 |
| small | 0 | **0.49 ± 0.06** | 0.97 ± 0.00 | 0.52 ± 0.03 | 0.07 ± 0.00 |

</details>

### Second half to first half

| model | ε | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|
| large | 0.05 | **0.16 ± 0.57** | 0.99 ± 0.00 | 0.00 ± 0.00 | 0.18 ± 0.00 |
| large | 0 | **0.43 ± 0.12** | 0.94 ± 0.03 | 0.21 ± 0.09 | 0.11 ± 0.00 |
| large, RoPE | 0.05 | **0.91 ± 0.04** | 0.97 ± 0.00 | 0.00 ± 0.00 | -1.12 ± 1.79 |
| small | 0.05 | **0.24 ± 0.48** | 0.98 ± 0.00 | 0.00 ± 0.00 | 0.17 ± 0.00 |
| small | 0 | **0.27 ± 0.22** | 0.86 ± 0.00 | -0.02 ± 0.07 | 0.10 ± 0.00 |

<details>
<summary>Process weighting</summary>

| model | ε | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|
| large | 0.05 | **0.25 ± 0.35** | 0.99 ± 0.00 | -0.01 ± 0.00 | 0.17 ± 0.00 |
| large | 0 | **0.43 ± 0.12** | 0.94 ± 0.03 | 0.22 ± 0.09 | 0.11 ± 0.00 |
| large, RoPE | 0.05 | **0.85 ± 0.08** | 0.98 ± 0.00 | -0.01 ± 0.00 | -6.54 ± 14.09 |
| small | 0.05 | **0.19 ± 0.37** | 0.99 ± 0.00 | -0.01 ± 0.00 | 0.16 ± 0.00 |
| small | 0 | **0.27 ± 0.22** | 0.86 ± 0.00 | -0.02 ± 0.07 | 0.10 ± 0.00 |

</details>

Did not reach the training-KL target: large, RoPE, ε = 0.05 (5 of 5 seeds); small, ε = 0.05 (5 of 5 seeds).
