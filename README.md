# belief-state

A small, self-contained PyTorch implementation of a transformer trained on the MRRXOR process, with a held-out linear probe from its residual stream to the Bayesian belief state. The setup follows

> A. S. Shai, S. E. Marzen, L. Teixeira, A. G. Oldenziel, and P. M. Riechers, *Transformers represent belief state geometry in their residual stream*, NeurIPS 2024, [arXiv:2405.15943](https://arxiv.org/abs/2405.15943).

## Results

We evaluate three models: a *large* one with the architecture of Shai et al. (2024), 4 layers and d_model 64, and a *small* one with 2 layers and d_model 32, each trained with 5 seeds for ε = 0.05 and 0; and the large one with rotary position embeddings (*RoPE*) instead of learned ones, trained with 5 seeds for ε = 0.05. All large-model runs, and the small-model runs with ε = 0, reached training KL 1e-4; the small-model runs with ε = 0.05 stopped at 100,000 steps, at training KL 4.8e-4 to 7.5e-4. The RoPE runs trained more slowly and stopped at 100,000 steps, at training KL 9.1e-4 to 9.6e-4.

Each entry is the held-out R² of a linear probe from the residual stream, as mean ± standard deviation over the 5 seeds. The main number, in bold, is for the belief state: the posterior over the five hidden states given the history. Three controls use the same features, fit, and held-out rows:

- **Next-token distribution:** the target is the Bayes-optimal next-token distribution, which the network is trained to output.
- **Shuffled belief state:** the distinct belief states are randomly permuted among themselves. Histories with the same belief still share a target, but the targets no longer reflect the geometry of the beliefs, so this measures the R² the probe gets for an arbitrary function of the belief. For ε = 0, there are only 35 distinct beliefs and the network separates them, so with the random split any relabeling of them is about as decodable as the beliefs themselves.
- **Untrained network:** the belief-state probe on the network at initialization.

The tables shown below weight histories *uniformly*: every sequence with positive probability counts equally. The collapsed tables use *process* weighting: trajectories are sampled from the process, so typical histories dominate.

### Random split

The probe is fitted on all positions of 80% of the trajectories and scored on all positions of the other 20%.

<!-- results random -->

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

<!-- /results -->

### Odd/even split

The probe is fitted on the odd positions (history lengths 1, 3, …, 31) of the training trajectories and scored on the even positions of the test trajectories. No history is then shared between fitting and scoring, and the probe has to carry over to positions it has not seen.

<!-- results parity -->

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

<!-- /results -->

### Halves split

The probe is fitted on one half of the positions of the training trajectories and scored on the other half of the test trajectories, so it has to carry over to positions farther from those it was fitted on.

Fitted on history lengths 1–16, scored on lengths 17–32:

<!-- results early -->

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

<!-- /results -->

Fitted on history lengths 17–32, scored on lengths 1–16:

<!-- results late -->

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

<!-- /results -->

## Setup

**Process.** MRRXOR (Mixing Random-Random-XOR) is a five-state hidden Markov model. It emits two independent random bits, then their XOR, and starts over. Before each emission, the hidden state jumps to one of the other four states with probability ε. With ε = 0 it is the RRXOR process of Crutchfield and Feldman (2003), studied by Shai et al. (2024). Belief states are computed exactly by Bayesian filtering from the stationary distribution.

**Why mixing.** With ε > 0, every binary sequence has positive probability, so a probe can be weighted uniformly over all histories. Without mixing, most sequences are impossible (any violated XOR), and the belief takes only 36 distinct values, including the initial one. With mixing, the beliefs form a rich, fractal set that a probe has to resolve. We run ε = 0.05, and ε = 0 for comparison with Shai et al. (2024).

**Models.** The large model is that of Shai et al. (2024), Appendix A.6: 4 layers, d_model 64, one attention head of dimension 8, ReLU MLPs of width 256, pre-LayerNorm, learned positional embeddings, causal attention, and no dropout (144k parameters). The small model has 2 layers, d_model 32, MLP width 128, and the same attention head (20k parameters). The RoPE model is the large one with rotary position embeddings (Su et al., 2021) on the queries and keys in place of the learned positional embeddings (142k parameters). Every weight matrix and embedding is drawn from N(0, σ²) with σ = 0.1; biases start at 0, and LayerNorm at gain 1 and bias 0.

**Context length 32.** Longer than the length-10 contexts of Shai et al. (2024), so there are many more distinct histories (2^32 sequences for ε > 0). The probe's training and test trajectories can then be numerous and disjoint.

**Training.** Next-token cross entropy, averaged over all 32 positions, on batches of 512 sequences sampled from the stationary process. Adam with learning rate 1e-3 (otherwise PyTorch defaults), linear warmup over 1,000 steps, then cosine decay to 1e-5 at step 100,000. Training stops when the *training KL* falls below 1e-4, or after 100,000 steps. The training KL is the KL divergence from the exact Bayes-optimal next-token distribution to the model's prediction, averaged over the last 500 batches. The same quantity on a fixed, separately sampled set of 4,096 sequences (the test KL) is logged every 1,000 steps. Shai et al. (2024) instead trained with SGD (learning rate 0.01, batch 64) for a fixed number of steps.

**Probe.** The features at each position are the residual stream at every layer as the network reads it: the LayerNorm-ed input of each block and the LayerNorm-ed output of the last block (the input to the unembedding), (layers + 1) × d_model features in all, 320 for the large model and 96 for the small one. Probing the raw residual stream after each layer instead, as described by Shai et al. (2024), changes the belief-state R² by at most 0.05 but lowers the next-token R² by up to 0.15 in the small model, since the logits are an affine function of the LayerNorm-ed output of the last block but not of the raw residual stream. One affine probe is *pooled* over all positions, as in Shai et al. (2024). We draw 8,192 trajectories of length 32 and split them 80/20 into training and test sets. R² is computed on held-out rows, pooled over the coordinates of the target; Shai et al. (2024) report in-sample fits, with a cross-validation check in their appendix. For the random split, the probe is fitted by least squares. For the odd/even and halves splits, least squares extrapolates badly to positions it has not seen, so the probe is a ridge regression. Its penalty, a multiple of the mean eigenvalue of the feature covariance, is chosen separately for each target by cross-validation over four groups of fitting positions of the training trajectories: {1, 9, 17, 25}, {3, 11, 19, 27}, … for the odd/even split, and four contiguous blocks of four positions for the halves.

## Files

| File | What it does |
|---|---|
| `mrrxor.py` | The process: exact transition tensor, sampling, Bayesian belief states, optimal next-token probabilities |
| `model.py` | The transformer, with the number of layers, d_model, and learned or rotary position embeddings as parameters |
| `train.py` | Training until the model's predictions are within a target KL of Bayes-optimal |
| `probe.py` | The held-out probes (random and odd/even splits) and their controls, and the test KL |
| `summarize.py` | Collects the probe results into the tables in `results.md` |
| `run_sweep.sh` | Trains and probes ε = 0.05 and 0 with 5 seeds each, for one model size |
| `tests.py` | Self-contained checks of the process, the model, the probe and the scripts |

The repository keeps one file per run, `runs/L<layers>_d<d_model>/eps<ε>/s<seed>.pt`: the final weights and the training summary (steps, training and test KL). The untrained network is regenerated from the seed. The run's log (`.log`), probe results (`.json`) and a checkpoint every 5,000 steps (in a directory of the same name) stay local.

Requirements: Python 3.10+ and PyTorch (tested with 2.14), plus NumPy for the tests. Training uses Apple's MPS or CUDA when available.

## Reproduce

```sh
./run_sweep.sh         # large model: trains 2 x 5 runs in parallel, probes them, writes results.md
./run_sweep.sh 2 32    # small model
for seed in 0 1 2 3 4; do python train.py --rope --eps 0.05 --seed $seed --out runs/L4_d64_rope/eps0.05/s$seed; done
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
