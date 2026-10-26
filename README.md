# belief-state

A small, self-contained PyTorch implementation of a transformer trained on the MRRXOR process, with a held-out linear probe from its residual stream to the Bayesian belief state. The setup follows

> A. S. Shai, S. E. Marzen, L. Teixeira, A. G. Oldenziel, and P. M. Riechers, *Transformers represent belief state geometry in their residual stream*, NeurIPS 2024, [arXiv:2405.15943](https://arxiv.org/abs/2405.15943).

## Results

We evaluate four models: a *large* one with the architecture of Shai et al. (2024), 4 layers and d_model 64, a *small* one with 2 layers and d_model 32, and both with rotary position embeddings (*RoPE*) instead of learned ones. Each is trained with 5 seeds for 100,000 steps, for ε = 0.05 and 0, and every entry is the mean ± standard deviation over the seeds. The *test KL* is the KL divergence from the Bayes-optimal next-token distribution to the model's at the end of training.

The other entries are held-out R² values of linear probes from the residual stream. The main number, in bold, is for the belief state: the posterior over the five hidden states given the history. Three controls use the same features, fit, and held-out rows:

- **Next-token distribution:** the target is the Bayes-optimal next-token distribution, which the network is trained to output.
- **Shuffled belief state:** the distinct belief states are randomly permuted among themselves. Histories with the same belief still share a target, but the targets no longer reflect the geometry of the beliefs, so this measures the R² the probe gets for an arbitrary function of the belief. For ε = 0, there are only 36 distinct beliefs and the network separates them, so with the random split any relabeling of them is about as decodable as the beliefs themselves.
- **Untrained network:** the belief-state probe on the network at initialization.

The tables shown below weight histories *uniformly*: every sequence with positive probability counts equally. The collapsed tables use *process* weighting: trajectories are sampled from the process, so typical histories dominate.

### Random split

The probe is fitted on all positions of 80% of the trajectories and scored on all positions of the other 20%.

<!-- results random -->

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 6.1e-5 ± 3.0e-5 | **0.93 ± 0.03** | 1.00 ± 0.00 | 0.13 ± 0.01 | 0.24 ± 0.01 |
| large | 0 | 1.1e-6 ± 1.7e-7 | **0.88 ± 0.05** | 1.00 ± 0.00 | 0.83 ± 0.04 | 0.12 ± 0.00 |
| large, RoPE | 0.05 | 9.5e-5 ± 2.5e-5 | **0.98 ± 0.00** | 1.00 ± 0.00 | 0.15 ± 0.00 | 0.32 ± 0.01 |
| large, RoPE | 0 | 4.5e-6 ± 2.8e-6 | **0.93 ± 0.03** | 1.00 ± 0.00 | 0.83 ± 0.03 | 0.19 ± 0.01 |
| small | 0.05 | 8.9e-4 ± 8.0e-4 | **0.69 ± 0.10** | 0.97 ± 0.01 | 0.07 ± 0.01 | 0.20 ± 0.00 |
| small | 0 | 2.0e-4 ± 3.5e-4 | **0.54 ± 0.09** | 0.94 ± 0.04 | 0.48 ± 0.06 | 0.10 ± 0.00 |
| small, RoPE | 0.05 | 1.9e-3 ± 4.5e-4 | **0.74 ± 0.04** | 0.98 ± 0.00 | 0.07 ± 0.01 | 0.26 ± 0.01 |
| small, RoPE | 0 | 1.6e-3 ± 3.0e-4 | **0.65 ± 0.03** | 0.92 ± 0.01 | 0.52 ± 0.02 | 0.14 ± 0.01 |

<details>
<summary>Process weighting</summary>

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 6.1e-5 ± 3.0e-5 | **0.92 ± 0.04** | 1.00 ± 0.00 | 0.16 ± 0.00 | 0.19 ± 0.00 |
| large | 0 | 1.1e-6 ± 1.7e-7 | **0.88 ± 0.05** | 1.00 ± 0.00 | 0.83 ± 0.04 | 0.12 ± 0.00 |
| large, RoPE | 0.05 | 9.5e-5 ± 2.5e-5 | **0.97 ± 0.01** | 1.00 ± 0.00 | 0.17 ± 0.01 | 0.27 ± 0.00 |
| large, RoPE | 0 | 4.5e-6 ± 2.8e-6 | **0.93 ± 0.03** | 1.00 ± 0.00 | 0.83 ± 0.03 | 0.19 ± 0.01 |
| small | 0.05 | 8.9e-4 ± 8.0e-4 | **0.65 ± 0.11** | 0.98 ± 0.01 | 0.11 ± 0.01 | 0.16 ± 0.00 |
| small | 0 | 2.0e-4 ± 3.5e-4 | **0.54 ± 0.09** | 0.94 ± 0.04 | 0.48 ± 0.06 | 0.10 ± 0.00 |
| small, RoPE | 0.05 | 1.9e-3 ± 4.5e-4 | **0.70 ± 0.03** | 0.97 ± 0.01 | 0.11 ± 0.01 | 0.21 ± 0.01 |
| small, RoPE | 0 | 1.6e-3 ± 3.0e-4 | **0.65 ± 0.03** | 0.92 ± 0.01 | 0.52 ± 0.02 | 0.14 ± 0.01 |

</details>

<!-- /results -->

### Odd/even split

The probe is fitted on the odd positions (history lengths 1, 3, …, 31) of the training trajectories and scored on the even positions (0, 2, …, 30) of the test trajectories. No history is then shared between fitting and scoring, and the probe has to carry over to positions it has not seen.

<!-- results parity -->

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 6.1e-5 ± 3.0e-5 | **0.83 ± 0.04** | 1.00 ± 0.00 | -0.02 ± 0.00 | 0.16 ± 0.00 |
| large | 0 | 1.1e-6 ± 1.7e-7 | **0.69 ± 0.09** | 0.98 ± 0.01 | 0.44 ± 0.07 | 0.08 ± 0.00 |
| large, RoPE | 0.05 | 9.5e-5 ± 2.5e-5 | **0.95 ± 0.02** | 1.00 ± 0.00 | -0.01 ± 0.00 | -0.22 ± 0.81 |
| large, RoPE | 0 | 4.5e-6 ± 2.8e-6 | **0.85 ± 0.04** | 0.99 ± 0.01 | 0.51 ± 0.03 | 0.08 ± 0.02 |
| small | 0.05 | 8.9e-4 ± 8.0e-4 | **0.60 ± 0.10** | 0.97 ± 0.01 | -0.01 ± 0.00 | 0.16 ± 0.01 |
| small | 0 | 2.0e-4 ± 3.5e-4 | **0.45 ± 0.11** | 0.92 ± 0.04 | 0.29 ± 0.04 | 0.08 ± 0.00 |
| small, RoPE | 0.05 | 1.9e-3 ± 4.5e-4 | **0.64 ± 0.09** | 0.97 ± 0.01 | 0.00 ± 0.00 | -1.98 ± 4.07 |
| small, RoPE | 0 | 1.6e-3 ± 3.0e-4 | **0.60 ± 0.04** | 0.91 ± 0.01 | 0.20 ± 0.24 | -1.19 ± 2.45 |

<details>
<summary>Process weighting</summary>

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 6.1e-5 ± 3.0e-5 | **0.81 ± 0.06** | 1.00 ± 0.00 | -0.04 ± 0.03 | 0.14 ± 0.00 |
| large | 0 | 1.1e-6 ± 1.7e-7 | **0.69 ± 0.09** | 0.98 ± 0.01 | 0.44 ± 0.07 | 0.08 ± 0.00 |
| large, RoPE | 0.05 | 9.5e-5 ± 2.5e-5 | **0.94 ± 0.03** | 1.00 ± 0.00 | -0.08 ± 0.03 | -0.20 ± 0.78 |
| large, RoPE | 0 | 4.5e-6 ± 2.8e-6 | **0.85 ± 0.04** | 0.99 ± 0.01 | 0.51 ± 0.03 | 0.08 ± 0.02 |
| small | 0.05 | 8.9e-4 ± 8.0e-4 | **0.55 ± 0.11** | 0.98 ± 0.01 | -0.01 ± 0.02 | 0.14 ± 0.00 |
| small | 0 | 2.0e-4 ± 3.5e-4 | **0.45 ± 0.11** | 0.92 ± 0.04 | 0.29 ± 0.04 | 0.08 ± 0.00 |
| small, RoPE | 0.05 | 1.9e-3 ± 4.5e-4 | **0.61 ± 0.06** | 0.97 ± 0.00 | -0.05 ± 0.06 | -1.44 ± 3.00 |
| small, RoPE | 0 | 1.6e-3 ± 3.0e-4 | **0.60 ± 0.04** | 0.91 ± 0.01 | 0.20 ± 0.24 | -1.19 ± 2.46 |

</details>

<!-- /results -->

### Halves split

The probe is fitted on one half of the positions of the training trajectories and scored on the other half of the test trajectories, so it has to carry over to positions farther from those it was fitted on.

Fitted on history lengths 0–15, scored on lengths 16–31:

<!-- results early -->

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 6.1e-5 ± 3.0e-5 | **0.71 ± 0.12** | 0.99 ± 0.00 | -0.01 ± 0.00 | 0.15 ± 0.00 |
| large | 0 | 1.1e-6 ± 1.7e-7 | **0.56 ± 0.13** | 0.98 ± 0.01 | 0.54 ± 0.02 | 0.07 ± 0.00 |
| large, RoPE | 0.05 | 9.5e-5 ± 2.5e-5 | **0.90 ± 0.05** | 1.00 ± 0.00 | -0.01 ± 0.00 | 0.16 ± 0.00 |
| large, RoPE | 0 | 4.5e-6 ± 2.8e-6 | **0.73 ± 0.09** | 0.99 ± 0.01 | 0.59 ± 0.10 | 0.07 ± 0.00 |
| small | 0.05 | 8.9e-4 ± 8.0e-4 | **0.48 ± 0.08** | 0.96 ± 0.02 | -0.01 ± 0.00 | 0.15 ± 0.00 |
| small | 0 | 2.0e-4 ± 3.5e-4 | **0.35 ± 0.06** | 0.95 ± 0.03 | 0.37 ± 0.02 | 0.07 ± 0.00 |
| small, RoPE | 0.05 | 1.9e-3 ± 4.5e-4 | **0.61 ± 0.05** | 0.97 ± 0.00 | 0.00 ± 0.00 | 0.16 ± 0.00 |
| small, RoPE | 0 | 1.6e-3 ± 3.0e-4 | **0.47 ± 0.06** | 0.92 ± 0.01 | 0.41 ± 0.03 | 0.07 ± 0.00 |

<details>
<summary>Process weighting</summary>

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 6.1e-5 ± 3.0e-5 | **0.67 ± 0.15** | 0.99 ± 0.00 | -0.01 ± 0.01 | 0.12 ± 0.00 |
| large | 0 | 1.1e-6 ± 1.7e-7 | **0.56 ± 0.13** | 0.98 ± 0.01 | 0.54 ± 0.02 | 0.07 ± 0.00 |
| large, RoPE | 0.05 | 9.5e-5 ± 2.5e-5 | **0.87 ± 0.08** | 1.00 ± 0.00 | -0.01 ± 0.00 | 0.13 ± 0.00 |
| large, RoPE | 0 | 4.5e-6 ± 2.8e-6 | **0.73 ± 0.09** | 0.99 ± 0.01 | 0.59 ± 0.10 | 0.07 ± 0.00 |
| small | 0.05 | 8.9e-4 ± 8.0e-4 | **0.40 ± 0.08** | 0.98 ± 0.02 | -0.02 ± 0.01 | 0.12 ± 0.00 |
| small | 0 | 2.0e-4 ± 3.5e-4 | **0.35 ± 0.06** | 0.95 ± 0.03 | 0.37 ± 0.02 | 0.07 ± 0.00 |
| small, RoPE | 0.05 | 1.9e-3 ± 4.5e-4 | **0.53 ± 0.04** | 0.96 ± 0.01 | -0.01 ± 0.00 | 0.13 ± 0.00 |
| small, RoPE | 0 | 1.6e-3 ± 3.0e-4 | **0.47 ± 0.06** | 0.92 ± 0.01 | 0.41 ± 0.03 | 0.07 ± 0.00 |

</details>

<!-- /results -->

Fitted on history lengths 16–31, scored on lengths 0–15:

<!-- results late -->

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 6.1e-5 ± 3.0e-5 | **0.51 ± 0.33** | 0.99 ± 0.00 | 0.00 ± 0.00 | 0.18 ± 0.01 |
| large | 0 | 1.1e-6 ± 1.7e-7 | **0.17 ± 0.20** | 0.92 ± 0.04 | 0.13 ± 0.14 | 0.11 ± 0.00 |
| large, RoPE | 0.05 | 9.5e-5 ± 2.5e-5 | **-546.25 ± 287.63** | -25.10 ± 20.89 | 0.00 ± 0.00 | -158.94 ± 172.34 |
| large, RoPE | 0 | 4.5e-6 ± 2.8e-6 | **-230.55 ± 232.43** | -5.83 ± 10.57 | -162.56 ± 173.19 | -126.47 ± 167.96 |
| small | 0.05 | 8.9e-4 ± 8.0e-4 | **0.48 ± 0.04** | 0.98 ± 0.01 | 0.00 ± 0.00 | 0.17 ± 0.01 |
| small | 0 | 2.0e-4 ± 3.5e-4 | **-0.36 ± 1.49** | 0.83 ± 0.02 | -0.10 ± 0.46 | 0.10 ± 0.00 |
| small, RoPE | 0.05 | 1.9e-3 ± 4.5e-4 | **-192.55 ± 121.97** | -6.10 ± 3.69 | 0.00 ± 0.00 | -43.35 ± 48.41 |
| small, RoPE | 0 | 1.6e-3 ± 3.0e-4 | **-103.07 ± 143.37** | -3.52 ± 6.67 | -42.15 ± 65.40 | -31.79 ± 40.27 |

<details>
<summary>Process weighting</summary>

| model | ε | test KL | belief state | next-token distribution | shuffled belief state | untrained network |
|---|---:|---:|---:|---:|---:|---:|
| large | 0.05 | 6.1e-5 ± 3.0e-5 | **0.41 ± 0.52** | 0.99 ± 0.00 | -0.02 ± 0.00 | 0.17 ± 0.00 |
| large | 0 | 1.1e-6 ± 1.7e-7 | **0.17 ± 0.20** | 0.92 ± 0.04 | 0.13 ± 0.14 | 0.11 ± 0.00 |
| large, RoPE | 0.05 | 9.5e-5 ± 2.5e-5 | **-590.84 ± 327.79** | -13.72 ± 3.12 | -0.02 ± 0.00 | -189.28 ± 282.79 |
| large, RoPE | 0 | 4.5e-6 ± 2.8e-6 | **-230.27 ± 232.45** | -6.05 ± 10.89 | -167.92 ± 177.63 | -126.09 ± 167.49 |
| small | 0.05 | 8.9e-4 ± 8.0e-4 | **0.47 ± 0.05** | 0.98 ± 0.01 | -0.02 ± 0.00 | 0.16 ± 0.01 |
| small | 0 | 2.0e-4 ± 3.5e-4 | **-0.36 ± 1.49** | 0.83 ± 0.02 | -0.10 ± 0.46 | 0.10 ± 0.00 |
| small, RoPE | 0.05 | 1.9e-3 ± 4.5e-4 | **-265.36 ± 229.37** | -1.42 ± 3.26 | -0.02 ± 0.00 | -27.94 ± 24.71 |
| small, RoPE | 0 | 1.6e-3 ± 3.0e-4 | **-102.90 ± 142.99** | -3.50 ± 6.64 | -42.13 ± 65.37 | -31.90 ± 40.45 |

</details>

<!-- /results -->

This direction fails for the RoPE models. Their features change mostly over the first few tokens and, in most runs, little after about eight, so a probe fitted on lengths 16–31 extrapolates far outside its range at lengths below about six, where almost all of its error arises.

## Setup

**Process.** MRRXOR (Mixing Random-Random-XOR) is a five-state hidden Markov model. It emits two independent random bits, then their XOR, and starts over. Before each emission, the hidden state jumps to one of the other four states with probability ε. With ε = 0 it is the RRXOR process of Crutchfield and Feldman (2003), studied by Shai et al. (2024). Belief states are computed exactly by Bayesian filtering from the stationary distribution.

**Why mixing.** With ε > 0, every binary sequence has positive probability, so a probe can be weighted uniformly over all histories. Without mixing, most sequences are impossible (any violated XOR), and the belief takes only 36 distinct values, including the initial one. With mixing, the beliefs form a rich, fractal set that a probe has to resolve. We run ε = 0.05, and ε = 0 for comparison with Shai et al. (2024).

**Models.** The large model is that of Shai et al. (2024), Appendix A.6: 4 layers, d_model 64, one attention head of dimension 8, ReLU MLPs of width 256, pre-LayerNorm, learned positional embeddings, causal attention, and no dropout (144k parameters). The small model has 2 layers, d_model 32, MLP width 128, and the same attention head (20k parameters). The RoPE models are the large and small ones with rotary position embeddings (Su et al., 2021) on the queries and keys in place of the learned positional embeddings (142k and 19k parameters). Every weight matrix and embedding is drawn from N(0, σ²) with σ = 0.1; biases start at 0, and LayerNorm at gain 1 and bias 0.

**Context length 32.** Longer than the length-10 contexts of Shai et al. (2024), so there are many more distinct histories (2^32 sequences for ε > 0). The probe's training and test trajectories can then be numerous and disjoint.

**Training.** The network reads a beginning-of-sequence (BOS) token followed by the first 31 tokens of a sequence, so that its 32 outputs predict all 32 tokens, the first included, and together give the probability of the whole sequence. Shai et al. (2024) used no BOS token, so their first output predicts the second token. The loss is the cross entropy of each token given the ones before it, averaged over the 32 tokens, on batches of 512 sequences sampled from the stationary process. Adam with learning rate 1e-3 (otherwise PyTorch defaults), linear warmup over 1,000 steps, then cosine decay to 1e-5 at step 100,000, where training stops. The *test KL*, the KL divergence from the exact Bayes-optimal next-token distribution to the model's prediction averaged over all tokens of a fixed, separately sampled set of 4,096 sequences, is logged every 1,000 steps. Shai et al. (2024) instead trained with SGD (learning rate 0.01, batch 64).

**Probe.** The features at each position are the residual stream at every layer as the network reads it: the LayerNorm-ed input of each block and the LayerNorm-ed output of the last block (the input to the unembedding), (layers + 1) × d_model features in all, 320 for the large model and 96 for the small one. Position n has read BOS and the first n tokens, n = 0, …, 31, and its target is the belief after those n tokens; at n = 0 this is the stationary distribution, the same in every trajectory. Probing the raw residual stream after each layer instead, as described by Shai et al. (2024), changes the belief-state R² by at most 0.11, but lowers the next-token R² by up to 0.2 in the small models (at most 0.02 in the large ones), since the logits are an affine function of the LayerNorm-ed output of the last block but not of the raw residual stream. One affine probe is *pooled* over all positions, as in Shai et al. (2024). We draw 8,192 trajectories of length 32 and split them 80/20 into training and test sets. R² is computed on held-out rows, pooled over the coordinates of the target; Shai et al. (2024) report in-sample fits, with a cross-validation check in their appendix. For the random split, the probe is fitted by least squares. For the odd/even and halves splits, least squares extrapolates badly to positions it has not seen, so the probe is a ridge regression. Its penalty, a multiple of the mean eigenvalue of the feature covariance, is chosen separately for each target by cross-validation over four groups of fitting positions of the training trajectories: {1, 9, 17, 25}, {3, 11, 19, 27}, … for the odd/even split, and four contiguous blocks of four positions for the halves.

## Files

| File | What it does |
|---|---|
| `mrrxor.py` | The process: exact transition tensor, sampling, Bayesian belief states, optimal next-token probabilities |
| `model.py` | The transformer, with the number of layers, d_model, and learned or rotary position embeddings as parameters |
| `train.py` | Training for a fixed number of steps, logging the KL from Bayes-optimal predictions to the model's |
| `probe.py` | The held-out probes (random, odd/even and halves splits) and their controls, and the test KL |
| `summarize.py` | Collects the probe results into the tables in `results.md` |
| `run_sweep.sh` | Trains and probes ε = 0.05 and 0 with 5 seeds each (or the seeds in `SEEDS`), for one model size and kind of position embedding |
| `tests.py` | Self-contained checks of the process, the model, the probe and the scripts |

The repository keeps one file per run, `runs/L<layers>_d<d_model>[_rope]/eps<ε>/s<seed>.pt`: the final weights and the training summary (steps, test KL). The untrained network is regenerated from the seed. The run's log (`.log`), probe results (`.json`) and a checkpoint every 5,000 steps (in a directory of the same name) stay local.

Requirements: Python 3.10+, PyTorch (tested with 2.14), and Numba (tested with 0.68) with NumPy. Training uses Apple's MPS or CUDA when available, with the model compiled by `torch.compile`.

## Reproduce

```sh
./run_sweep.sh              # large model: trains 2 x 5 runs in parallel, probes them, writes results.md
./run_sweep.sh 2 32         # small model
./run_sweep.sh 4 64 rope    # large model with RoPE
./run_sweep.sh 2 32 rope    # small model with RoPE
```

To recreate the tables from the stored checkpoints:

```sh
python probe.py runs/*/*/*.pt    # writes runs/<model>/eps<ε>/s<seed>.json
python summarize.py              # writes results.md
```

To follow a run while it trains, `python probe.py runs/<model>/eps<ε>/s<seed> --checkpoint latest` probes the newest checkpoint, and `--checkpoint all` probes every checkpoint.

## Tests

```sh
python tests.py    # about half a minute
```

The tests compare the process, its belief states and its optimal predictions with a brute-force sum over hidden-state paths. They check that the model is causal and returns the stated features, that the probe pairs each position with the right belief, splits the rows and weights histories as described, and they run the scripts end to end on a tiny model.
