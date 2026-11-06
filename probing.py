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
the two halves. All penalties are selected by leaving out one fitting position at a time, fitting on the others,
and measuring weighted mean squared prediction error at the omitted position. We choose the penalty with the
lowest average error across positions, using only the training trajectories.
The chosen penalty is used for a final fit to all training trajectories at the fitting lengths. The outer test
trajectories, and any lengths excluded from fitting, are used only for scoring.

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

"""
from dataclasses import dataclass

import torch

import mrrxor
from model import Transformer, untrained


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


def position_grams(x, y, weights):
    """Weighted Gram matrices of [1, features, targets], one per position.

    x and y have shape (training trajectories, positions, coordinates); weights has one entry per trajectory.
    These statistics can be reused for overlapping sets of fitting positions.
    """
    grams = []
    for position in range(x.shape[1]):
        features, targets = x[:, position], y[:, position]
        rows = torch.cat((torch.ones_like(features[:, :1]), features, targets), 1)
        rows *= weights.sqrt()[:, None]
        grams.append(rows.T @ rows)
    return torch.stack(grams)


def fit_ridge_cv(grams, n_features, target_sizes):
    """Choose alpha by leave-one-position-out weighted MSE, then refit on all supplied positions.

    Each target block, described by target_sizes, selects its own alpha. The per-position Gram matrices
    suffice both to fit ridge and to evaluate validation error without predictions for every alpha.
    """
    if len(grams) < 2:
        raise ValueError("Leave-one-position-out validation needs at least two fitting positions.")
    total = grams.sum(0)
    cv_error = grams.new_zeros(len(ALPHAS), len(target_sizes))
    negative_identity = -torch.eye(sum(target_sizes), dtype=grams.dtype).expand(len(ALPHAS), -1, -1)
    for validation in grams:
        coef = ridge_path(total - validation, n_features)
        # [1, x, y] @ residual_map equals prediction minus target.
        residual_map = torch.cat((coef, negative_identity), 1)
        squared_error = (residual_map * (validation @ residual_map)).sum(1)
        errors = torch.stack([block.sum(1) for block in squared_error.split(target_sizes, 1)], 1)
        cv_error += errors / validation[0, 0]
    cv_error /= len(grams)
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


@dataclass
class ProbeData:
    """Shared trajectories, weights, targets and outer test mask for one weighting."""
    tokens: torch.Tensor
    weights: torch.Tensor
    targets: dict[str, torch.Tensor]
    test: torch.Tensor


def prepare_data(T: torch.Tensor, context: int, count: int = 8192,
                 test_fraction: float = 0.2, seed: int = 0) -> dict[str, ProbeData]:
    """Prepare both weightings once for reuse across the trained and untrained networks."""
    test = split(count, test_fraction, seed)
    if not (~test).any() or not test.any():
        raise ValueError("Need at least one training trajectory and one test trajectory.")
    data = {}
    for weighting in ("uniform", "process"):
        tokens, weights = trajectories(T, context, weighting, count, seed)
        beliefs = mrrxor.beliefs(T, tokens)[:, 1:-1]  # belief after n tokens, n = 1, ..., context - 1
        targets = {"": beliefs, "next_": mrrxor.next_token_probabilities(T, beliefs),
                   "shuffled_": shuffled(beliefs, seed)}
        data[weighting] = ProbeData(tokens, weights.double(), targets, test)
    return data


def probe(model: Transformer, data: dict[str, ProbeData]) -> dict:
    """Score the belief state and controls on all four splits using a shared prepared dataset."""
    out = {}
    lengths = torch.arange(1, model.cfg.context)
    all_positions = lengths > 0
    first_half = lengths < model.cfg.context // 2
    splits = {"": (all_positions, all_positions), "parity_": (lengths % 2 == 1, lengths % 2 == 0),
              "early_": (first_half, ~first_half), "late_": (~first_half, first_half)}
    for weighting, prepared in data.items():
        tokens, weights, targets, test = prepared.tokens, prepared.weights, prepared.targets, prepared.test
        with torch.no_grad():
            features = torch.cat([model(chunk, features=True)[1] for chunk in tokens.split(1024)]).double()
        features = features[:, 1:]  # history lengths 1, ..., context - 1
        row_weights = weights[:, None].expand(-1, len(lengths))
        stacked = torch.cat(list(targets.values()), -1)
        sizes = [value.shape[-1] for value in targets.values()]
        grams = position_grams(features[~test], stacked[~test], weights[~test])
        for split_name, (fit, evaluate) in splits.items():
            coef, alphas = fit_ridge_cv(grams[fit], features.shape[-1], sizes)
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


def evaluate(model: Transformer, training: dict, settings: dict, include_initial: bool) -> dict:
    """Compute test KL and probe scores, sharing prepared data with the optional initialization baseline."""
    T = mrrxor.transition_tensor(training["eps"])
    test_tokens, test_optimal = mrrxor.test_set(T, model.cfg.context)
    with torch.no_grad():
        test_kl = mrrxor.kl_to_model(test_optimal, model(test_tokens))
    data = prepare_data(T, model.cfg.context, **settings)
    result = {"test_kl": test_kl, **probe(model, data)}
    if include_initial:
        initial = untrained(model.cfg, training["init_std"], training["seed"]).eval()
        result.update({f"init_{key}": value for key, value in probe(initial, data).items()})
    return result


