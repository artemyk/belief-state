"""Pooled held-out linear probe from the transformer's residual stream to the Bayesian belief state.

As in Shai et al. (2024), one linear probe is shared by all positions. We draw trajectories of length 32 (the
model's context), run the model on each, and take the activations at every position: the LayerNorm-ed input of each
block and the LayerNorm-ed output of the last block ((n_layers + 1) x d_model features, 5 x 64 = 320 by default).
The target at position n is the exact belief state after the first n tokens. The trajectories are split 80/20, and
the probe is scored with
R^2 = 1 - weighted sum of squared errors / weighted sum of squared deviations from the weighted mean, pooled over
the target's coordinates. Two splits of the rows:
  random split    an affine least-squares probe fitted on all positions of the training trajectories and scored on
                  all positions of the test trajectories (in probe.json also for each position separately)
  odd/even split  (keys prefixed parity_) fitted on the odd positions (history lengths 1, 3, ..., 31) of the training
                  trajectories and scored on the even positions of the test trajectories, so that no history is
                  shared and the probe must carry over to positions it has not seen. Least squares extrapolates badly
                  to unseen positions, so this probe is a ridge regression. Its penalty is alpha times the mean
                  eigenvalue of the weighted feature covariance, with alpha chosen from ALPHAS separately for each
                  target, by leave-one-group-out cross-validation over four groups of odd positions ({1, 9, 17, 25},
                  {3, 11, 19, 27}, ...) of the training trajectories (keys parity_*alpha_*).

Histories are weighted in two ways, through the trajectory sample (used for both fitting and scoring):
  process  trajectories sampled from the process
  uniform  trajectories uniform over all sequences with positive probability. If every binary string has positive
           probability (eps > 0), these are uniformly random strings. Otherwise (RRXOR, eps = 0) they are sampled from
           the process and weighted by 1 / probability.
Every trajectory contributes one row per position, so all positions carry equal total weight. Short prefixes recur
across trajectories, so at the first few positions most test rows coincide with training rows.

Two controls use the same features and splits (one solve fits all targets):
  next_      the Bayes-optimal next-token distribution, which the network is trained to output
  shuffled_  the belief states with the distinct beliefs randomly permuted among themselves: histories that share a
             belief still share a target, but the targets no longer reflect the geometry of the beliefs. This measures
             how much R^2 the probe gets for an arbitrary function of the belief: for eps > 0 mostly at short prefixes
             that occur in both the training and test trajectories, for eps = 0 because the network separates the 35
             distinct beliefs.

Also reports the test KL of the checkpoint: the mean KL divergence from the Bayes-optimal next-token distribution
to the model's, over all positions of the fixed held-out set of 4,096 sequences used in train.py.

Usage: python probe.py runs/<model>/<run> [more runs] [--checkpoint final|latest|all|step_<step>]
  final: probes final.pt and, as a baseline, the untrained network (regenerated from the run's seed), and writes
         probe.json
  step_<step>: probes that checkpoint and writes probe_<name>.json
  latest: the newest step_<step>.pt; all: every step_<step>.pt not yet probed
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


def weighted_fit(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    """Weighted minimum-norm least squares with an intercept. The activations are float32, so singular values below
    1e-6 of the largest are treated as zero; otherwise rounding noise in nearly degenerate directions (e.g. at small
    initialization) gets large coefficients."""
    w = w / w.sum()
    x_mean, y_mean = w @ x, w @ y
    root = w.sqrt()[:, None]
    coef = torch.linalg.lstsq(root * (x - x_mean), root * (y - y_mean), rcond=1e-6, driver="gelsd").solution
    return lambda x_new: (x_new - x_mean) @ coef + y_mean


ALPHAS = torch.logspace(-8, 1, 19, dtype=torch.float64)


def ridge_path(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> list:
    """Weighted ridge regressions with an unpenalized intercept, one predictor per alpha in ALPHAS. The penalty is
    alpha times the mean eigenvalue of the weighted feature covariance."""
    w = w / w.sum()
    x_mean, y_mean = w @ x, w @ y
    root = w.sqrt()[:, None]
    xc = root * (x - x_mean)
    values, vectors = torch.linalg.eigh(xc.T @ xc)
    values = values.clamp_min(0)
    scale = values.mean().clamp_min(torch.finfo(values.dtype).tiny)  # constant features get zero coefficients
    projected = vectors.T @ (xc.T @ (root * (y - y_mean)))
    coefs = [vectors @ (projected / (values + alpha * scale)[:, None]) for alpha in ALPHAS]
    return [lambda x_new, coef=coef: (x_new - x_mean) @ coef + y_mean for coef in coefs]


def weighted_r2(y: torch.Tensor, prediction: torch.Tensor, w: torch.Tensor) -> float:
    w = w / w.sum()
    return float(1 - (w @ ((y - prediction) ** 2).sum(1)) / (w @ ((y - w @ y) ** 2).sum(1)))


def probe(model: Transformer, T: torch.Tensor, count: int = 8192, test_fraction: float = 0.2, seed: int = 0) -> dict:
    """Pooled held-out R^2 for both weightings and both splits (by position too, for the random split), of the
    belief state and the two controls."""
    out = {}
    context = model.cfg.context
    for weighting in ("uniform", "process"):
        tokens, weights = trajectories(T, context, weighting, count, seed)
        with torch.no_grad():
            features = torch.cat([model(chunk, features=True)[1] for chunk in tokens.split(1024)]).double()
        beliefs = mrrxor.beliefs(T, tokens)[:, 1:]  # belief after each prefix: (count, length, 5)
        test = split(count, test_fraction, seed)
        row_weights = weights.double()[:, None].expand(-1, context)
        targets = {"": beliefs, "next_": mrrxor.next_token_probabilities(T, beliefs),
                   "shuffled_": shuffled(beliefs, seed)}
        stacked = torch.cat(list(targets.values()), -1)
        predict = weighted_fit(features[~test].flatten(0, 1), stacked[~test].flatten(0, 1),
                               row_weights[~test].flatten())
        predictions = predict(features[test].flatten(0, 1)).view(int(test.sum()), context, -1)
        for (prefix, target), prediction in zip(targets.items(),
                                                predictions.split([t.shape[-1] for t in targets.values()], -1)):
            out[f"{prefix}r2_{weighting}"] = weighted_r2(target[test].flatten(0, 1), prediction.flatten(0, 1),
                                                         row_weights[test].flatten())
            out[f"{prefix}r2_by_position_{weighting}"] = [
                weighted_r2(target[test][:, n], prediction[:, n], row_weights[test][:, n]) for n in range(context)]

        # odd/even split: ridge penalty per target by leave-one-group-out over odd positions of training trajectories
        position = torch.arange(context)
        odd, group = position % 2 == 0, (position // 2) % 4  # index 0 is history length 1
        train, evaluate = ~test[:, None] & odd, test[:, None] & ~odd
        sizes = [t.shape[-1] for t in targets.values()]
        cv = torch.zeros(len(targets), len(ALPHAS))
        for g in range(4):
            fit_rows, val_rows = train & (group != g), train & (group == g)
            for k, predict in enumerate(ridge_path(features[fit_rows], stacked[fit_rows], row_weights[fit_rows])):
                for j, (target, prediction) in enumerate(zip(targets.values(),
                                                             predict(features[val_rows]).split(sizes, -1))):
                    cv[j, k] += weighted_r2(target[val_rows], prediction, row_weights[val_rows]) / 4
        predictors = ridge_path(features[train], stacked[train], row_weights[train])
        for j, (prefix, target) in enumerate(targets.items()):
            k = int(cv[j].argmax())
            prediction = predictors[k](features[evaluate]).split(sizes, -1)[j]
            out[f"parity_{prefix}r2_{weighting}"] = weighted_r2(target[evaluate], prediction, row_weights[evaluate])
            out[f"parity_{prefix}alpha_{weighting}"] = float(ALPHAS[k])
    return out


def load(path: Path) -> tuple[Transformer, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Transformer(Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def checkpoint_names(run: Path, checkpoint: str) -> list[str]:
    steps = sorted(path.stem for path in run.glob("step_*.pt"))
    if checkpoint == "latest":
        return steps[-1:]
    if checkpoint == "all":
        return [name for name in steps if not (run / f"probe_{name}.json").exists()]
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

    for run, name in ((run, name) for run in args.runs for name in checkpoint_names(run, args.checkpoint)):
        model, checkpoint = load(run / f"{name}.pt")
        meta = checkpoint["args"]
        T = mrrxor.transition_tensor(float(meta["eps"]))
        test_tokens, test_optimal = mrrxor.test_set(T, model.cfg.context)
        with torch.no_grad():
            test_kl = mrrxor.kl_to_model(test_optimal, model(test_tokens))
        result = {"layers": model.cfg.n_layers, "d_model": model.cfg.d_model, "eps": float(meta["eps"]),
                  "init_std": float(meta["init_std"]), "seed": int(meta["seed"]), "steps": checkpoint["step"],
                  "test_kl": test_kl, **probe(model, T, **settings)}
        if name == "final":
            result.update(train_kl=checkpoint["train_kl"], reached_target=checkpoint["reached_target"])
            initial = untrained(model.cfg, float(meta["init_std"]), int(meta["seed"])).eval()
            result.update({f"init_{key}": value for key, value in probe(initial, T, **settings).items()})
            (run / "probe.json").write_text(json.dumps(result, indent=2))
        else:
            (run / f"probe_{name}.json").write_text(json.dumps(result, indent=2))
        print(f"{run.parent.name}/{run.name} ({name}, step {result['steps']}): test KL {test_kl:.2e}, pooled held-out "
              f"R^2 {result['r2_uniform']:.3f} (uniform), {result['r2_process']:.3f} (process); next-token "
              f"{result['next_r2_uniform']:.3f}, {result['next_r2_process']:.3f}; shuffled "
              f"{result['shuffled_r2_uniform']:.3f}, {result['shuffled_r2_process']:.3f}"
              + (f"; untrained {result['init_r2_uniform']:.3f}, {result['init_r2_process']:.3f}"
                 if name == "final" else ""), flush=True)


if __name__ == "__main__":
    main()
