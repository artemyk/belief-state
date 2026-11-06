"""Pooled held-out linear probe from the transformer's residual stream to the Bayesian belief state.

As in Shai et al. (2024), one linear probe is shared by all positions. We draw trajectories of length 32 (the
model's context), run the model on each, and take the activations at every position: the LayerNorm-ed input of each
block and the LayerNorm-ed output of the last block ((n_layers + 1) x d_model features, 5 x 64 = 320 by default).
The target at position n is the exact belief state after the first n tokens, n = 1, ..., 31. Position 0, which reads
only the BOS token, is not probed: it has no history, and its belief is the stationary distribution in every
trajectory. The trajectories are split 80/20, and
the probe is scored by weighted R^2, pooled over the target coordinates.
All splits use an affine ridge regression. The random split fits and scores all positive history lengths. The
parity_ split fits odd lengths and scores even lengths, early_ fits 1–15 and scores 16–31, and late_ reverses
the two halves. Random-split penalties minimize weighted validation squared error over five folds of whole
training trajectories. Held-out-position penalties maximize mean validation R^2 over four groups of fitting
positions, as in the original protocol: interleaved odd lengths for parity_, contiguous blocks for early_ and late_.
The chosen penalty is used for a final fit to all training trajectories at the fitting lengths. The outer test
trajectories and scoring lengths are used only for scoring.

Histories are weighted in two ways, through the trajectory sample (used for both fitting and scoring):
  process  trajectories sampled from the process
  uniform  trajectories uniform over all sequences with positive probability. If every binary string has positive
           probability (eps > 0), these are uniformly random strings. Otherwise (RRXOR, eps = 0) they are sampled from
           the process and weighted by 1 / probability.
Every trajectory contributes one row per position, so all positions carry equal total weight. Short prefixes recur
across trajectories, so at the first few positions most test rows coincide with training rows.

Two controls use the same features and splits:
  next_      the Bayes-optimal next-token distribution, which the network is trained to output
  shuffled_  the belief states with the distinct beliefs randomly permuted among themselves: histories that share a
             belief still share a target, but the targets no longer reflect the geometry of the beliefs. This measures
             how much R^2 the probe gets for an arbitrary function of the belief: for eps > 0 mostly at short prefixes
             that occur in both the training and test trajectories, for eps = 0 because the network separates the 35
             distinct beliefs.

Also reports the test KL of the checkpoint: the mean KL divergence from the Bayes-optimal next-token distribution
to the model's, over all positions of the fixed held-out set of 4,096 sequences used in train.py.

Usage: python probe.py runs/<model>/eps<eps>/s<seed>.pt [more runs] [--checkpoint final|latest|all|step_<step>]
  A run can also be given without the .pt extension.
  final: probes <run>.pt and, as a baseline, the untrained network (regenerated from the run's seed), and writes
         <run>.json
  step_<step>: probes <run>/step_<step>.pt and writes <run>/probe_step_<step>.json
  latest: the newest step checkpoint; all: every step checkpoint not yet probed
"""
import argparse
import json
from pathlib import Path

import torch

import mrrxor
from model import Config, Transformer, untrained

def trajectories(T: torch.Tensor, length: int, weighting: str, count: int,
                 seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Token sequences (count, length) and their weights under the given weighting."""
    generator = torch.Generator().manual_seed(seed)
    if weighting == "uniform" and (T.sum(2) > 0).all():  # every binary string has positive probability
        return torch.randint(0, 2, (count, length), generator=generator), torch.ones(count)
    tokens = mrrxor.sample(T, count, length, generator)
    if weighting == "process":
        return tokens, torch.ones(count)
    return tokens, 1 / mrrxor.sequence_probabilities(T, tokens)


def split(count: int, test_fraction: float, seed: int) -> torch.Tensor:
    """Boolean mask (count,) of the test trajectories: a random subset of round(test_fraction * count) of them."""
    test = torch.zeros(count, dtype=torch.bool)
    test[torch.randperm(count, generator=torch.Generator().manual_seed(seed))[: round(test_fraction * count)]] = True
    return test


def shuffled(beliefs: torch.Tensor, seed: int) -> torch.Tensor:
    """The beliefs with the distinct belief states (rounded to 1e-10) randomly permuted among themselves."""
    values, index = torch.unique(beliefs.flatten(0, 1).round(decimals=10), dim=0, return_inverse=True)
    permutation = torch.randperm(len(values), generator=torch.Generator().manual_seed(seed))
    return values[permutation][index].view_as(beliefs)


ALPHAS = torch.logspace(-8, 1, 19, dtype=torch.float64)
N_FOLDS = 5


def ridge_path(gram: torch.Tensor, n_features: int) -> torch.Tensor:
    """Affine coefficients for each alpha. gram is the weighted Gram matrix of [1, features, targets]."""
    moment = gram / gram[0, 0]
    mean = moment[0, 1:]
    cov = moment[1:, 1:] - torch.outer(mean, mean)
    values, vectors = torch.linalg.eigh(cov[:n_features, :n_features])
    values = values.clamp_min(0)
    scale = values.mean()
    slopes = gram.new_zeros(len(ALPHAS), n_features, len(mean) - n_features)
    if scale > 0:
        projected = vectors.T @ cov[:n_features, n_features:]
        slopes = vectors @ (projected / (values + ALPHAS[:, None] * scale)[:, :, None])
    intercepts = mean[n_features:] - mean[:n_features] @ slopes
    return torch.cat((intercepts[:, None, :], slopes), 1)


def fit_ridge_cv(x, y, weights, folds, target_sizes, *, hold_out_positions=False):
    """Choose alpha by trajectory CV (pooled MSE) or position CV (mean R²), then refit on all fitting rows."""
    # A fold's Gram matrix suffices both to fit ridge and to evaluate validation error without
    # materializing predictions for every alpha. Position folds contain all training trajectories.
    n_features = x.shape[-1]
    grams = []
    for fold in folds:
        if hold_out_positions:
            features, targets = x[:, fold].flatten(0, 1), y[:, fold].flatten(0, 1)
            row_weights = weights.repeat_interleave(len(fold))
        else:
            features, targets = x[fold].flatten(0, 1), y[fold].flatten(0, 1)
            row_weights = weights[fold].repeat_interleave(x.shape[1])
        rows = torch.cat((torch.ones_like(features[:, :1]), features, targets), 1)
        rows *= row_weights.sqrt()[:, None]
        grams.append(rows.T @ rows)
    total = torch.stack(grams).sum(0)
    cv_error = x.new_zeros(len(ALPHAS), len(target_sizes))
    negative_identity = -torch.eye(y.shape[-1], dtype=x.dtype).expand(len(ALPHAS), -1, -1)
    for validation in grams:
        coef = ridge_path(total - validation, n_features)
        # [1, x, y] @ residual_map equals prediction minus target.
        residual_map = torch.cat((coef, negative_identity), 1)
        squared_error = (residual_map * (validation @ residual_map)).sum(1)
        errors = torch.stack([block.sum(1) for block in squared_error.split(target_sizes, 1)], 1)
        if hold_out_positions:
            target_second = validation.diagonal()[1 + n_features:]
            target_total = validation[0, 1 + n_features:]
            centered = target_second - target_total.square() / validation[0, 0]
            variation = torch.stack([block.sum() for block in centered.split(target_sizes)])
            scale = torch.stack([block.abs().sum() for block in target_second.split(target_sizes)])
            valid = variation > 32 * torch.finfo(x.dtype).eps * scale
            # R² is undefined in a constant-target fold, so that fold does not select its target's alpha.
            errors[:, valid] /= variation[valid]
            errors[:, ~valid] = 0
        cv_error += errors
    best = cv_error.argmin(0).tolist()
    final = ridge_path(total, n_features)
    coefficients = torch.cat([block[k] for block, k in zip(final.split(target_sizes, 2), best)], 1)
    return coefficients, [float(ALPHAS[k]) for k in best]


def weighted_r2(y: torch.Tensor, prediction: torch.Tensor, w: torch.Tensor) -> float | None:
    """R² is undefined for a constant target (e.g. the next-bit probability after one observation)."""
    if (y - y[0]).abs().max() <= 32 * torch.finfo(y.dtype).eps * y.abs().max():
        return None
    w = w / w.sum()
    return float(1 - (w @ ((y - prediction) ** 2).sum(1)) / (w @ ((y - w @ y) ** 2).sum(1)))


def probe(model: Transformer, T: torch.Tensor, count: int = 8192, test_fraction: float = 0.2, seed: int = 0) -> dict:
    """Pooled held-out R^2 for both weightings and all four splits (by position too, for the random split), of the
    belief state and the two controls."""
    out = {}
    context = model.cfg.context
    for weighting in ("uniform", "process"):
        tokens, weights = trajectories(T, context, weighting, count, seed)
        with torch.no_grad():
            features = torch.cat([model(chunk, features=True)[1] for chunk in tokens.split(1024)]).double()
        features = features[:, 1:]  # history lengths 1, ..., context - 1
        beliefs = mrrxor.beliefs(T, tokens)[:, 1:-1]  # belief after n tokens, n = 1, ..., context - 1
        lengths = torch.arange(1, context)
        test = split(count, test_fraction, seed)
        row_weights = weights.double()[:, None].expand(-1, len(lengths))
        targets = {"": beliefs, "next_": mrrxor.next_token_probabilities(T, beliefs),
                   "shuffled_": shuffled(beliefs, seed)}
        stacked = torch.cat(list(targets.values()), -1)
        fit_count = int((~test).sum())
        if fit_count < N_FOLDS or not test.any():
            raise ValueError("Need at least five training trajectories and one test trajectory.")
        folds = torch.randperm(fit_count, generator=torch.Generator().manual_seed(seed)).tensor_split(N_FOLDS)
        all_positions = lengths > 0
        first_half = lengths < context // 2
        splits = {"": (all_positions, all_positions), "parity_": (lengths % 2 == 1, lengths % 2 == 0),
                  "early_": (first_half, ~first_half), "late_": (~first_half, first_half)}
        sizes = [value.shape[-1] for value in targets.values()]
        for split_name, (fit, evaluate) in splits.items():
            cv_folds = folds
            if split_name:
                groups = ((lengths // 2) % 4 if split_name == "parity_" else
                          (lengths % (context // 2)) // max(1, context // 8))
                cv_folds = [torch.where(groups[fit] == group)[0] for group in groups[fit].unique()]
            coef, alphas = fit_ridge_cv(features[~test][:, fit], stacked[~test][:, fit],
                                        weights[~test].double(), cv_folds, sizes,
                                        hold_out_positions=bool(split_name))
            predictions = features[test][:, evaluate] @ coef[1:] + coef[0]
            w = row_weights[test][:, evaluate]
            for (prefix, target), pred, alpha in zip(targets.items(), predictions.split(sizes, -1), alphas):
                key = split_name + prefix
                actual = target[test][:, evaluate]
                out[f"{key}r2_{weighting}"] = weighted_r2(actual.flatten(0, 1), pred.flatten(0, 1), w.flatten())
                out[f"{key}alpha_{weighting}"] = alpha
                if not split_name:
                    out[f"{key}r2_by_position_{weighting}"] = [None] + [
                        weighted_r2(actual[:, i], pred[:, i], w[:, i]) for i in range(len(lengths))]
    return out


def load(path: Path) -> tuple[Transformer, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Transformer(Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def files(run: Path, name: str) -> tuple[Path, Path]:
    """The checkpoint and the results file of a run (named without extension) for a checkpoint name."""
    if name == "final":
        return Path(f"{run}.pt"), Path(f"{run}.json")
    return run / f"{name}.pt", run / f"probe_{name}.json"


def checkpoint_names(run: Path, checkpoint: str) -> list[str]:
    steps = sorted(path.stem for path in run.glob("step_*.pt"))
    if checkpoint == "latest":
        return steps[-1:]
    if checkpoint == "all":
        return [name for name in steps if not files(run, name)[1].exists()]
    return [checkpoint]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--trajectories", type=int, default=8192)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    settings = dict(count=args.trajectories, test_fraction=args.test_fraction, seed=args.seed)

    runs = [Path(str(run).removesuffix(".pt")) for run in args.runs]
    for run, name in ((run, name) for run in runs for name in checkpoint_names(run, args.checkpoint)):
        checkpoint_path, results_path = files(run, name)
        model, checkpoint = load(checkpoint_path)
        meta = checkpoint["args"]
        T = mrrxor.transition_tensor(float(meta["eps"]))
        test_tokens, test_optimal = mrrxor.test_set(T, model.cfg.context)
        with torch.no_grad():
            test_kl = mrrxor.kl_to_model(test_optimal, model(test_tokens))
        result = {"layers": model.cfg.n_layers, "d_model": model.cfg.d_model, "rope": model.cfg.rope,
                  "eps": float(meta["eps"]), "init_std": float(meta["init_std"]), "seed": int(meta["seed"]),
                  "steps": checkpoint["step"], "test_kl": test_kl,
                  "probe_settings": {**settings, "trajectory_folds": N_FOLDS, "relative_alphas": ALPHAS.tolist(),
                                     "random_cv": "five trajectory folds, pooled weighted MSE",
                                     "held_out_position_cv": "four fitting-position groups, mean weighted R2"},
                  **probe(model, T, **settings)}
        if name == "final":
            initial = untrained(model.cfg, float(meta["init_std"]), int(meta["seed"])).eval()
            result.update({f"init_{key}": value for key, value in probe(initial, T, **settings).items()})
        results_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"{run.parent.parent.name}/{run.parent.name}/{run.name} ({name}, step {result['steps']}): "
              f"test KL {test_kl:.2e}, pooled held-out R^2 {result['r2_uniform']:.3f} (uniform), "
              f"{result['r2_process']:.3f} (process); next-token "
              f"{result['next_r2_uniform']:.3f}, {result['next_r2_process']:.3f}; shuffled "
              f"{result['shuffled_r2_uniform']:.3f}, {result['shuffled_r2_process']:.3f}"
              + (f"; untrained {result['init_r2_uniform']:.3f}, {result['init_r2_process']:.3f}"
                 if name == "final" else ""), flush=True)


if __name__ == "__main__":
    main()
