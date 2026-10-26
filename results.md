### Random split

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 5.0e-5 | **0.95** | 1.00 | 0.13 | 0.24 |
| large | 0 | 1.1e-6 | **0.92** | 1.00 | 0.87 | 0.12 |
| large, RoPE | 0.05 | 8.4e-5 | **0.98** | 1.00 | 0.15 | 0.33 |
| large, RoPE | 0 | 3.1e-6 | **0.95** | 1.00 | 0.85 | 0.19 |
| small | 0.05 | 4.6e-4 | **0.74** | 0.98 | 0.07 | 0.20 |
| small | 0 | 6.4e-6 | **0.56** | 0.96 | 0.52 | 0.10 |
| small, RoPE | 0.05 | 1.8e-3 | **0.70** | 0.97 | 0.06 | 0.26 |
| small, RoPE | 0 | 1.8e-3 | **0.61** | 0.92 | 0.49 | 0.14 |

<details>
<summary>Process weighting</summary>

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 5.0e-5 | **0.95** | 1.00 | 0.16 | 0.19 |
| large | 0 | 1.1e-6 | **0.92** | 1.00 | 0.87 | 0.12 |
| large, RoPE | 0.05 | 8.4e-5 | **0.98** | 1.00 | 0.18 | 0.27 |
| large, RoPE | 0 | 3.1e-6 | **0.95** | 1.00 | 0.85 | 0.19 |
| small | 0.05 | 4.6e-4 | **0.70** | 0.99 | 0.11 | 0.16 |
| small | 0 | 6.4e-6 | **0.56** | 0.96 | 0.52 | 0.10 |
| small, RoPE | 0.05 | 1.8e-3 | **0.67** | 0.98 | 0.10 | 0.21 |
| small, RoPE | 0 | 1.8e-3 | **0.61** | 0.92 | 0.49 | 0.14 |

</details>

### Odd/even split

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 5.0e-5 | **0.86** | 1.00 | -0.01 | 0.17 |
| large | 0 | 1.1e-6 | **0.79** | 0.99 | 0.43 | 0.09 |
| large, RoPE | 0.05 | 8.4e-5 | **0.97** | 1.00 | -0.01 | -1.67 |
| large, RoPE | 0 | 3.1e-6 | **0.90** | 0.99 | 0.52 | 0.07 |
| small | 0.05 | 4.6e-4 | **0.67** | 0.98 | -0.01 | 0.16 |
| small | 0 | 6.4e-6 | **0.37** | 0.95 | 0.29 | 0.08 |
| small, RoPE | 0.05 | 1.8e-3 | **0.51** | 0.97 | 0.00 | -0.10 |
| small, RoPE | 0 | 1.8e-3 | **0.57** | 0.92 | 0.34 | -0.12 |

<details>
<summary>Process weighting</summary>

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 5.0e-5 | **0.86** | 1.00 | -0.07 | 0.14 |
| large | 0 | 1.1e-6 | **0.79** | 0.99 | 0.43 | 0.09 |
| large, RoPE | 0.05 | 8.4e-5 | **0.96** | 1.00 | -0.08 | -1.60 |
| large, RoPE | 0 | 3.1e-6 | **0.90** | 0.99 | 0.52 | 0.07 |
| small | 0.05 | 4.6e-4 | **0.63** | 0.99 | 0.00 | 0.14 |
| small | 0 | 6.4e-6 | **0.37** | 0.95 | 0.29 | 0.08 |
| small, RoPE | 0.05 | 1.8e-3 | **0.54** | 0.97 | -0.02 | -0.11 |
| small, RoPE | 0 | 1.8e-3 | **0.57** | 0.92 | 0.34 | -0.12 |

</details>

### First half to second half

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 5.0e-5 | **0.81** | 0.99 | -0.01 | 0.15 |
| large | 0 | 1.1e-6 | **0.78** | 0.99 | 0.58 | 0.07 |
| large, RoPE | 0.05 | 8.4e-5 | **0.93** | 1.00 | -0.01 | 0.16 |
| large, RoPE | 0 | 3.1e-6 | **0.73** | 0.99 | 0.68 | 0.07 |
| small | 0.05 | 4.6e-4 | **0.51** | 0.97 | -0.01 | 0.15 |
| small | 0 | 6.4e-6 | **0.31** | 0.97 | 0.36 | 0.07 |
| small, RoPE | 0.05 | 1.8e-3 | **0.59** | 0.96 | 0.00 | 0.16 |
| small, RoPE | 0 | 1.8e-3 | **0.54** | 0.92 | 0.38 | 0.07 |

<details>
<summary>Process weighting</summary>

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 5.0e-5 | **0.83** | 0.99 | -0.01 | 0.12 |
| large | 0 | 1.1e-6 | **0.78** | 0.99 | 0.58 | 0.07 |
| large, RoPE | 0.05 | 8.4e-5 | **0.92** | 1.00 | -0.01 | 0.13 |
| large, RoPE | 0 | 3.1e-6 | **0.73** | 0.99 | 0.68 | 0.07 |
| small | 0.05 | 4.6e-4 | **0.38** | 0.99 | -0.01 | 0.13 |
| small | 0 | 6.4e-6 | **0.31** | 0.97 | 0.36 | 0.07 |
| small, RoPE | 0.05 | 1.8e-3 | **0.54** | 0.96 | -0.01 | 0.13 |
| small, RoPE | 0 | 1.8e-3 | **0.54** | 0.92 | 0.38 | 0.07 |

</details>

### Second half to first half

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 5.0e-5 | **0.69** | 0.99 | 0.00 | 0.18 |
| large | 0 | 1.1e-6 | **0.51** | 0.95 | 0.25 | 0.11 |
| large, RoPE | 0.05 | 8.4e-5 | **-116.03** | -7.94 | 0.00 | -62.76 |
| large, RoPE | 0 | 3.1e-6 | **-2.89** | 0.05 | -1.90 | -62.37 |
| small | 0.05 | 4.6e-4 | **0.50** | 0.98 | 0.00 | 0.18 |
| small | 0 | 6.4e-6 | **-3.02** | 0.85 | -0.92 | 0.11 |
| small, RoPE | 0.05 | 1.8e-3 | **-55.98** | -0.39 | 0.00 | -62.34 |
| small, RoPE | 0 | 1.8e-3 | **-352.19** | -1.00 | -156.58 | -86.12 |

<details>
<summary>Process weighting</summary>

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 5.0e-5 | **0.75** | 0.99 | -0.02 | 0.17 |
| large | 0 | 1.1e-6 | **0.51** | 0.95 | 0.25 | 0.11 |
| large, RoPE | 0.05 | 8.4e-5 | **-744.22** | -8.49 | -0.02 | -135.36 |
| large, RoPE | 0 | 3.1e-6 | **-2.89** | 0.15 | -1.88 | -62.45 |
| small | 0.05 | 4.6e-4 | **0.47** | 0.99 | -0.02 | 0.17 |
| small | 0 | 6.4e-6 | **-3.02** | 0.85 | -0.91 | 0.11 |
| small, RoPE | 0.05 | 1.8e-3 | **-177.80** | -0.08 | -0.02 | -69.20 |
| small, RoPE | 0 | 1.8e-3 | **-351.32** | -1.00 | -156.50 | -86.70 |

</details>

Seeds per row: 1.
