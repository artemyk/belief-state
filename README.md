# belief-state

A small, self-contained PyTorch implementation of a transformer trained on the MRRXOR process, with a held-out linear probe from its residual stream to the Bayesian belief state. The setup follows

> A. S. Shai, S. E. Marzen, L. Teixeira, A. G. Oldenziel, and P. M. Riechers, *Transformers represent belief state geometry in their residual stream*, NeurIPS 2024, [arXiv:2405.15943](https://arxiv.org/abs/2405.15943).

## Results

We compare large (4 layers, d_model 64) and small (2 layers, d_model 32) transformers with rotary or learned absolute position embeddings. The default is RoPE with a standard beginning-of-sequence (BOS) token. Each model is trained with 5 seeds for 100,000 steps, for ε = 0.05 and 0.

For each ε, the first table gives the *test KL*, the KL divergence from the Bayes-optimal next-token distribution to the model's at the end of training, and the held-out R² of a linear probe from the residual stream to the belief state, the posterior over the five hidden states given the history. The probe is scored on four splits of its rows:

- **Random:** fitted on all positions of 80% of the trajectories and scored on all positions of the other 20%.
- **Odd/even:** fitted on the odd positions (history lengths 1, 3, …, 31) of the training trajectories and scored on the even positions (2, 4, …, 30) of the test trajectories. No history is then shared between fitting and scoring, and the probe has to carry over to positions it has not seen.
- **1st → 2nd half:** fitted on history lengths 1–15 and scored on lengths 16–31, so that the probe has to carry over to positions farther from those it was fitted on. **2nd → 1st half** is the reverse.

The second table gives three controls for the random split, which use the same features, fit, and held-out rows; those for the other splits are collapsed below it:

- **Next-token distribution:** the target is the Bayes-optimal next-token distribution, which the network is trained to output.
- **Shuffled belief state:** the distinct belief states are randomly permuted among themselves. Histories with the same belief still share a target, but the targets no longer reflect the geometry of the beliefs, so this measures the R² the probe gets for an arbitrary function of the belief. For ε = 0, there are only 35 distinct beliefs and the network separates them, so with the random split any relabeling of them is about as decodable as the beliefs themselves.
- **Untrained network:** the belief-state probe on the network at initialization.

Entries are means over the seeds, with the standard deviation for the belief state. The tables weight histories *uniformly*: every sequence with positive probability counts equally. The collapsed tables use *process* weighting: trajectories are sampled from the process, so typical histories dominate.

### ε = 0.05

<!-- results eps0.05 -->

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

<!-- /results -->

### ε = 0

<!-- results eps0 -->

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

<!-- /results -->

The late-to-early split tests extrapolation across history lengths. At ε = 0, RRXOR synchronizes: after a few tokens the history usually determines the hidden state, and the mean belief entropy falls from 1.33 nats after one token to 0.05 after 16. At ε = 0.05, the corresponding entropies are 1.33 and 0.68 nats (8,192 process trajectories, seed 0). The distribution of residual activations also changes with length. A low late-to-early R² measures how the fitted readout transfers across these changes. A negative value means it does worse than predicting the test-set mean belief.

## Setup

**Process.** MRRXOR (Mixing Random-Random-XOR) is a five-state hidden Markov model. It emits two independent random bits, then their XOR, and starts over. Before each emission, the hidden state jumps to one of the other four states with probability ε. With ε = 0 it is the RRXOR process of Crutchfield and Feldman (2003), studied by Shai et al. (2024). Belief states are computed exactly by Bayesian filtering from the stationary distribution.

**Mixing probabilities.** We run ε = 0.05 and ε = 0. With ε = 0.05, every binary sequence has positive probability, and the beliefs form a fractal set. With ε = 0, supported histories obey the XOR constraint, and the belief takes 36 distinct values, including the initial one.

**Models.** The large model is that of Shai et al. (2024), Appendix A.6: 4 layers, d_model 64, one attention head of dimension 8, ReLU MLPs of width 256, pre-LayerNorm, learned positional embeddings, causal attention, and no dropout (144k parameters). The small model has 2 layers, d_model 32, MLP width 128, and the same attention head (20k parameters). The RoPE models are the large and small ones with rotary position embeddings (Su et al., 2021) on the queries and keys in place of the learned positional embeddings (142k and 19k parameters). Every weight matrix and embedding is drawn from N(0, σ²) with σ = 0.1; biases start at 0, and LayerNorm at gain 1 and bias 0.

**Training.** The network reads a beginning-of-sequence (BOS) token followed by the first 31 tokens of a sequence, so that its 32 outputs predict all 32 tokens. The loss is the mean next-token cross entropy on batches of 512 sequences sampled from the stationary process.

We use Adam with learning rate 1e-3 (otherwise PyTorch defaults), linear warmup over 1,000 steps, then cosine decay to 1e-5 at step 100,000. Every 1,000 steps we log the *test KL*, the mean KL divergence from the exact Bayes-optimal predictions to the model's predictions on a fixed, separately sampled set of 4,096 sequences.

**Probe.** The features are the LayerNorm-ed input of each block and the LayerNorm-ed output of the last block, giving 320 features for the large model and 96 for the small one. At each history length n = 1, …, 31, the target is the Bayesian belief after the first n tokens.

We draw 8,192 trajectories of length 32 and split them 80/20 into training and test sets. For each split, we fit one affine ridge probe shared across its fitting positions, with an unpenalized intercept. R² is computed on the held-out rows, pooled over the coordinates of the target.

The ridge penalty is selected separately for each target and weighting. The 19 candidates are spaced by half-decades from 1e−8 to 10, expressed as multiples of the mean eigenvalue of the fitting-feature covariance. After selecting a penalty, we refit on all fitting rows.

For the **random split**, we minimize pooled, weighted validation squared error over five folds of whole training trajectories (probe seed 0).

For the **held-out-position splits**, we maximize mean weighted validation R² over four groups of fitting positions in the training trajectories. The odd/even groups are {1, 9, 17, 25}, {3, 11, 19, 27}, {5, 13, 21, 29}, and {7, 15, 23, 31}. The halves use contiguous blocks, {1, 2, 3}, {4, …, 7}, … for early fitting and {16, …, 19}, … for late fitting. In the late-to-early split, both fitting and penalty selection use lengths 16–31.

Per-position R² is undefined when the target is constant, such as the next-bit probability after a single observation. These entries are stored as `null`.

## Files

| File | What it does |
|---|---|
| `mrrxor.py` | The process: exact transition tensor, sampling, Bayesian belief states, optimal next-token probabilities |
| `model.py` | The transformer, with the number of layers, d_model, and learned or rotary position embeddings as parameters |
| `train.py` | Training for a fixed number of steps, logging the KL from Bayes-optimal predictions to the model's |
| `probe.py` | The held-out probes (random, odd/even and halves splits) and their controls, and the test KL |
| `summarize.py` | Collects the probe results into the tables in `results.md` |
| `run_sweep.sh` | Trains and probes ε = 0.05 and 0 with 5 seeds each (or the seeds in `SEEDS`), for one model size and kind of position embedding |
| `modal_runs.py` | Trains and probes runs on [Modal](https://modal.com) GPUs instead of locally |
| `tests.py` | Self-contained checks of the process, the model, the probe and the scripts |

Each run in `runs/L<layers>_d<d_model>[_rope]/eps<ε>/` includes the final checkpoint (`s<seed>.pt`) and its probe results (`s<seed>.json`), including the penalty chosen for each split, target, and weighting. The untrained network is regenerated from its seed. Logs and intermediate checkpoints stay local.

Requirements: Python 3.10+, PyTorch (tested with 2.14), and Numba (tested with 0.68) with NumPy. Training uses Apple's MPS or CUDA when available, with the model compiled by `torch.compile` (on CUDA, replayed as CUDA graphs).

## Reproduce

```sh
./run_sweep.sh              # large RoPE model, both mixing probabilities, five seeds
./run_sweep.sh 2 32         # small RoPE model
./run_sweep.sh 4 64 learned # large model with learned absolute position embeddings
./run_sweep.sh 2 32 learned # small model with learned absolute position embeddings
```

Alternatively, `modal run modal_runs.py::main` trains and probes all 40 runs on Modal GPUs; the file describes its options and how to copy the results back.

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
