"""Self-contained checks of the process, the training targets, the model, and the probe.

Run from the repository root with `python tests.py` (about half a minute). The process and its Bayesian
quantities are compared with a brute-force reference that enumerates hidden-state paths and is written independently
of mrrxor.py. The model is checked for causality and for the features it returns. The probe is checked for the
alignment of activations with beliefs, its two splits, its weighting, its ridge fit, and its controls. The last test
runs train.py, probe.py and summarize.py end to end on a tiny model.
"""
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import mrrxor
import probe
from model import Config, Transformer, rotary, untrained

ROOT = Path(__file__).resolve().parent
EPSILONS = (0.0, 0.05)

# RRXOR, written out by hand: (state, token) -> (next state, probability). States A..E are 0..4.
EDGES = {(0, 0): (1, 0.5), (0, 1): (2, 0.5),  # A: first random bit
         (1, 0): (4, 0.5), (1, 1): (3, 0.5),  # B (first bit 0): second bit; E emits the XOR 0, D emits 1
         (2, 0): (3, 0.5), (2, 1): (4, 0.5),  # C (first bit 1): second bit 0 -> XOR 1 (D), 1 -> XOR 0 (E)
         (3, 1): (0, 1.0), (4, 0): (0, 1.0)}  # D emits 1, E emits 0, both back to A


def reference_tensor(eps: float) -> np.ndarray:
    """P(token x, next state t | state s): jump to another state u with probability eps (uniformly), then emit."""
    T = np.zeros((2, 5, 5))
    for s, u in itertools.product(range(5), repeat=2):
        jump = 1 - eps if u == s else eps / 4
        for (state, x), (t, p) in EDGES.items():
            if state == u:
                T[x, s, t] += jump * p
    return T


def reference_stationary(T: np.ndarray) -> np.ndarray:
    P = T.sum(0)
    A = np.vstack([P.T - np.eye(5), np.ones(5)])
    return np.linalg.lstsq(A, np.r_[np.zeros(5), 1.0], rcond=None)[0]


def reference_joint(T: np.ndarray, x: tuple) -> np.ndarray:
    """P(x_1..x_k, s_k) for k = 0..len(x), by summing over every hidden path s_0..s_k (no recursion)."""
    pi = reference_stationary(T)
    out = []
    for k in range(len(x) + 1):
        paths = np.array(list(itertools.product(range(5), repeat=k + 1)))
        p = pi[paths[:, 0]]
        for j in range(k):
            p = p * T[x[j], paths[:, j], paths[:, j + 1]]
        out.append(np.bincount(paths[:, k], weights=p, minlength=5))
    return np.array(out)


class Process(unittest.TestCase):
    def test_transition_tensor_matches_hand_written_edges(self):
        for eps in EPSILONS:
            T = mrrxor.transition_tensor(eps)
            np.testing.assert_allclose(T.numpy(), reference_tensor(eps), atol=1e-15)
            self.assertTrue((T >= 0).all())
            np.testing.assert_allclose(T.sum((0, 2)).numpy(), np.ones(5), atol=1e-15)

    def test_stationary_distribution(self):
        for eps in EPSILONS:
            pi = mrrxor.stationary_distribution(mrrxor.transition_tensor(eps)).numpy()
            np.testing.assert_allclose(pi, reference_stationary(reference_tensor(eps)), atol=1e-12)
        np.testing.assert_allclose(mrrxor.stationary_distribution(mrrxor.transition_tensor(0.0)).numpy(),
                                   [1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6], atol=1e-12)

    def test_beliefs_and_probabilities_against_path_enumeration(self):
        """Beliefs after every prefix, sequence probabilities, and the optimal next-token distribution, for every
        sequence of length 5 with positive probability."""
        for eps in (0.0, 0.05):
            T, T_ref = mrrxor.transition_tensor(eps), reference_tensor(eps)
            total = 0.0
            for x in itertools.product((0, 1), repeat=5):
                joint = reference_joint(T_ref, x)
                probability = joint[-1].sum()
                if probability == 0:
                    self.assertEqual(eps, 0.0)
                    continue
                total += probability
                tokens = torch.tensor([x])
                np.testing.assert_allclose(mrrxor.sequence_probabilities(T, tokens).numpy(), [probability],
                                           rtol=1e-12)
                np.testing.assert_allclose(mrrxor.beliefs(T, tokens)[0].numpy(),
                                           joint / joint.sum(1, keepdims=True), atol=1e-12)
                # entry n: distribution of x_{n+1} given x_1..x_n = P(x_1..x_n, 1) / P(x_1..x_n)
                predicted = mrrxor.optimal_predictions(T, tokens)[0].numpy()
                for n in range(5):
                    one = (joint[n] @ T_ref[1]).sum() / joint[n].sum()
                    np.testing.assert_allclose(predicted[n], [1 - one, one], atol=1e-12)
            self.assertAlmostEqual(total, 1.0, places=12)

    def test_sampler_matches_sequence_probabilities(self):
        """Frequencies of all 16 length-4 strings in 200,000 samples (started from the stationary distribution)."""
        for eps in (0.0, 0.05):
            T = mrrxor.transition_tensor(eps)
            count = 200_000
            tokens = mrrxor.sample(T, count, 4, torch.Generator().manual_seed(0))
            codes = (tokens * torch.tensor([8, 4, 2, 1])).sum(1)
            frequency = (torch.bincount(codes, minlength=16).double() / count).tolist()
            for code in range(16):
                x = tuple(int(c) for c in f"{code:04b}")
                p = reference_joint(reference_tensor(eps), x)[-1].sum()
                self.assertLess(abs(frequency[code] - p), 5 * np.sqrt(p * (1 - p) / count) + 1e-9, (eps, x))

    def test_rrxor_samples_satisfy_xor_in_one_phase(self):
        tokens = mrrxor.sample(mrrxor.transition_tensor(0.0), 2000, 32, torch.Generator().manual_seed(1)).numpy()
        for x in tokens:
            self.assertTrue(any(all(x[i + 2] == x[i] ^ x[i + 1] for i in range(phase, 30, 3)) for phase in range(3)))

    def test_rrxor_has_36_beliefs_and_equal_sequence_probabilities_per_phase(self):
        """README claims: for eps = 0 the belief takes 36 distinct values (including the initial one), 5 of them
        vertices; a length-32 sequence has probability k 2^-22 / 3, where k = 1, 2 or 3 is the number of phases it is
        consistent with."""
        T = mrrxor.transition_tensor(0.0)
        prior = mrrxor.stationary_distribution(T)
        key = lambda b: tuple(np.round(b.numpy(), 10))
        seen, frontier = {key(prior)}, [prior]
        while frontier:
            new = []
            for b in frontier:
                for x in (0, 1):
                    u = b @ T[x]
                    if u.sum() > 0 and key(u / u.sum()) not in seen:
                        seen.add(key(u / u.sum()))
                        new.append(u / u.sum())
            frontier = new
        self.assertEqual(len(seen), 36)
        self.assertEqual(sum(max(b) > 1 - 1e-9 for b in seen), 5)
        tokens = mrrxor.sample(T, 20_000, 32, torch.Generator().manual_seed(2))
        k = mrrxor.sequence_probabilities(T, tokens) * 3 * 2.0 ** 22
        np.testing.assert_allclose(k.numpy(), k.round().numpy(), atol=1e-9)
        self.assertTrue(set(k.round().int().tolist()) <= {1, 2, 3})
        self.assertGreater(float((k.round() == 1).double().mean()), 0.99)

    def test_test_set_is_fixed_and_aligned(self):
        T = mrrxor.transition_tensor(0.05)
        tokens, optimal = mrrxor.test_set(T, 32, size=64)
        tokens2, optimal2 = mrrxor.test_set(T, 32, size=64)
        self.assertTrue(torch.equal(tokens, tokens2) and torch.equal(optimal, optimal2))
        self.assertTrue(torch.equal(optimal, mrrxor.optimal_predictions(T, tokens)))


class Training(unittest.TestCase):
    def test_optimal_predictions_match_empirical_next_tokens(self):
        """train.py pairs the logits at position n (which have seen tokens[:, :n]) with optimal_predictions[:, n] and
        the target tokens[:, n]. Check this pairing empirically: group sampled sequences by their first n tokens (none
        for n = 0) and compare the frequency of token n + 1 with the prediction."""
        T = mrrxor.transition_tensor(0.05)
        tokens = mrrxor.sample(T, 200_000, 4, torch.Generator().manual_seed(3))
        optimal = mrrxor.optimal_predictions(T, tokens)
        for n in range(4):
            prefixes = (tokens[:, :n] * 2 ** torch.arange(n)).sum(1)
            for code in prefixes.unique():
                rows = prefixes == code
                observed = float(tokens[rows, n].double().mean())
                self.assertTrue(torch.allclose(optimal[rows, n, 1], optimal[rows, n, 1][0]))
                expected = float(optimal[rows, n, 1][0])
                self.assertLess(abs(observed - expected), 5 * np.sqrt(0.25 / int(rows.sum())) + 1e-9)

    def test_kl(self):
        generator = torch.Generator().manual_seed(4)
        logits = torch.randn(100, 8, 2, generator=generator, dtype=torch.float64)
        p = torch.softmax(torch.randn(100, 8, 2, generator=generator, dtype=torch.float64), -1)
        p[:10, :, 0], p[:10, :, 1] = 0.0, 1.0  # deterministic predictions, as for eps = 0
        q = torch.softmax(logits, -1)
        expected = torch.where(p > 0, p * (p.clamp_min(1e-300).log() - q.log()), 0.0).sum(-1).mean()
        self.assertAlmostEqual(mrrxor.kl_to_model(p, logits), float(expected), places=12)
        self.assertAlmostEqual(mrrxor.kl_to_model(q, logits), 0.0, places=12)


class Model(unittest.TestCase):
    def setUp(self):
        self.model = untrained(Config(context=16, n_layers=2, d_model=32, d_mlp=128), 0.1, seed=0).eval()

    def test_causal(self):
        """Position n has seen exactly tokens[:, :n]: changing token k leaves the logits and features at positions up
        to k unchanged and changes position k + 1. Position 0 reads only BOS, so it is the same for every sequence."""
        tokens = torch.randint(0, 2, (8, 16), generator=torch.Generator().manual_seed(5))
        with torch.no_grad():
            logits, features = self.model(tokens, features=True)
            torch.testing.assert_close(features[:, 0], features[:1, 0].expand(8, -1))
            for k in range(16):
                changed = tokens.clone()
                changed[:, k] = 1 - changed[:, k]
                logits2, features2 = self.model(changed, features=True)
                torch.testing.assert_close(features2[:, :k + 1], features[:, :k + 1], rtol=0, atol=1e-6)
                torch.testing.assert_close(logits2[:, :k + 1], logits[:, :k + 1], rtol=0, atol=1e-6)
                if k < 15:
                    self.assertGreater(float((features2[:, k + 1] - features[:, k + 1]).abs().max()), 1e-3)

    def test_features(self):
        """(n_layers + 1) d_model features: the LayerNorm-ed input of each block, then the final LayerNorm output,
        which the unembedding maps to the logits."""
        m, tokens = self.model, torch.randint(0, 2, (4, 16), generator=torch.Generator().manual_seed(6))
        with torch.no_grad():
            logits, features = m(tokens, features=True)
            self.assertEqual(features.shape, (4, 16, 3 * 32))
            bos = torch.full((4, 1), m.cfg.vocab)
            x = m.embed(torch.cat([bos, tokens[:, :-1]], 1)) + m.pos_embed(torch.arange(16))
            for i, block in enumerate(m.blocks):
                torch.testing.assert_close(features[..., 32 * i: 32 * (i + 1)], block.ln1(x))
                x, _ = block(x)
            torch.testing.assert_close(features[..., 64:], m.ln_final(x))
            torch.testing.assert_close(m.unembed(features[..., 64:]), logits)

    def test_untrained_is_reproducible(self):
        torch.randn(10)  # the global RNG state beforehand does not matter
        a = untrained(Config(), 0.1, seed=3).state_dict()
        b = untrained(Config(), 0.1, seed=3).state_dict()
        c = untrained(Config(), 0.1, seed=4).state_dict()
        self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))
        self.assertFalse(torch.equal(a["embed.weight"], c["embed.weight"]))
        for key, value in a.items():
            if key.endswith("bias"):
                self.assertTrue((value == 0).all(), key)
            elif ".ln" in key or key.startswith("ln_final"):
                self.assertTrue((value == 1).all(), key)
            elif value.numel() > 1000:
                self.assertAlmostEqual(float(value.std()), 0.1, delta=0.01)

    def test_rope(self):
        """Rotary embeddings keep norms, make query-key products depend only on the relative position, and keep the
        model causal; the RoPE model has no learned positional embeddings."""
        generator = torch.Generator().manual_seed(10)
        q = torch.randn(1, 1, 1, 8, generator=generator).expand(1, 1, 32, 8)
        k = torch.randn(1, 1, 1, 8, generator=generator).expand(1, 1, 32, 8)
        torch.testing.assert_close(rotary(q).norm(dim=-1), q.norm(dim=-1))
        scores = (rotary(q) @ rotary(k).transpose(-1, -2))[0, 0]
        for offset in (0, 3, 11):
            diagonal = scores.diagonal(-offset)
            torch.testing.assert_close(diagonal, diagonal[0].expand_as(diagonal), rtol=0, atol=1e-5)
        model = untrained(Config(context=16, n_layers=2, d_model=32, d_mlp=128, rope=True), 0.1, seed=0).eval()
        self.assertIsNone(model.pos_embed)
        tokens = torch.randint(0, 2, (8, 16), generator=generator)
        with torch.no_grad():
            features = model(tokens, features=True)[1]
            changed = tokens.clone()
            changed[:, 9] = 1 - changed[:, 9]
            torch.testing.assert_close(model(changed, features=True)[1][:, :10], features[:, :10], rtol=0, atol=1e-6)

    def test_parameter_counts(self):
        """README: 144k parameters (Shai et al. 2024 architecture) and 20k (2 layers, d_model 32); 142k and 19k with
        RoPE."""
        count = lambda cfg: sum(p.numel() for p in Transformer(cfg).parameters())
        self.assertEqual(count(Config()), 144_418)
        self.assertEqual(count(Config(rope=True)), 142_370)
        self.assertEqual(count(Config(n_layers=2, d_model=32, d_mlp=128)), 20_370)
        self.assertEqual(count(Config(n_layers=2, d_model=32, d_mlp=128, rope=True)), 19_346)


class BeliefModel(torch.nn.Module):
    """Stand-in for the transformer whose features at position n are the exact belief after tokens[:, :n + lag]
    (zero-padded to the transformer's width). With lag = 0 the probe must give R^2 = 1; lag = 1 is an off-by-one.
    blank = "odd", "even", "first" or "second" zeroes the features at odd or even history lengths, or in the first or
    second half of the context."""

    def __init__(self, T: torch.Tensor, lag: int = 0, blank: str | None = None):
        super().__init__()
        self.T, self.lag, self.blank, self.cfg = T, lag, blank, Config()

    def forward(self, tokens: torch.Tensor, features: bool = False):
        b = mrrxor.beliefs(self.T, tokens)[:, self.lag: tokens.shape[1] + self.lag].float()
        if self.blank:
            positions = {"odd": slice(1, None, 2), "even": slice(0, None, 2), "first": slice(0, 16),
                         "second": slice(16, None)}  # index n is history length n
            b[:, positions[self.blank]] = 0
        padded = torch.cat([b, torch.zeros(*b.shape[:2], (self.cfg.n_layers + 1) * self.cfg.d_model - 5)], -1)
        return torch.zeros(*tokens.shape, 2), padded


class Probe(unittest.TestCase):
    def test_split_is_by_trajectory(self):
        test = probe.split(1000, 0.2, seed=0)
        self.assertEqual(int(test.sum()), 200)
        self.assertTrue(torch.equal(test, probe.split(1000, 0.2, seed=0)))
        self.assertFalse(torch.equal(test, probe.split(1000, 0.2, seed=1)))

    def test_weighted_fit_and_r2(self):
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(500, 10, generator=generator, dtype=torch.float64)
        w = torch.rand(500, generator=generator, dtype=torch.float64)
        y = x @ torch.randn(10, 3, generator=generator, dtype=torch.float64) + torch.tensor([1.0, -2.0, 0.5])
        self.assertAlmostEqual(probe.weighted_r2(y, probe.weighted_fit(x, y, w)(x), w), 1.0, places=10)
        ones = torch.ones(500, dtype=torch.float64)
        noisy = y + torch.randn(500, 3, generator=generator, dtype=torch.float64)
        prediction = probe.weighted_fit(x, noisy, ones)(x)
        expected = 1 - ((noisy - prediction) ** 2).sum() / ((noisy - noisy.mean(0)) ** 2).sum()
        self.assertAlmostEqual(probe.weighted_r2(noisy, prediction, ones), float(expected), places=12)
        self.assertLess(probe.weighted_r2(noisy, torch.zeros_like(noisy) + noisy.mean(0), ones), 1e-12)

    def test_ridge_path(self):
        """The smallest penalty reproduces least squares; larger penalties shrink the predictions toward the mean."""
        generator = torch.Generator().manual_seed(9)
        x = torch.randn(500, 10, generator=generator, dtype=torch.float64)
        w = torch.rand(500, generator=generator, dtype=torch.float64)
        y = x @ torch.randn(10, 3, generator=generator, dtype=torch.float64) + torch.randn(500, 3, generator=generator,
                                                                                           dtype=torch.float64)
        predictors = probe.ridge_path(x, y, w)
        torch.testing.assert_close(predictors[0](x), probe.weighted_fit(x, y, w)(x), rtol=0, atol=1e-6)
        spread = [float(p(x).std(0).sum()) for p in predictors]
        self.assertTrue(all(a >= b for a, b in zip(spread, spread[1:])))
        self.assertLess(spread[-1], 0.5 * spread[0])

    def test_uniform_weighting(self):
        """eps > 0: uniformly random strings. eps = 0: process samples weighted by 1 / probability, which makes every
        possible sequence equally likely."""
        tokens, weights = probe.trajectories(mrrxor.transition_tensor(0.05), 32, "uniform", 1000, seed=0)
        self.assertTrue((weights == 1).all() and set(tokens.unique().tolist()) == {0, 1})
        T = mrrxor.transition_tensor(0.0)
        tokens, weights = probe.trajectories(T, 6, "uniform", 200_000, seed=0)
        support = [x for x in itertools.product((0, 1), repeat=6)
                   if reference_joint(reference_tensor(0.0), x)[-1].sum() > 0]
        codes = (tokens * 2 ** torch.arange(5, -1, -1)).sum(1)
        mass = torch.zeros(64, dtype=torch.float64).index_add_(0, codes, weights.double()) / weights.sum()
        for x in support:
            code = int("".join(map(str, x)), 2)
            self.assertAlmostEqual(float(mass[code]), 1 / len(support), delta=0.15 / len(support))
        self.assertAlmostEqual(float(mass.sum()), 1.0, places=12)

    def test_shuffled_permutes_distinct_beliefs(self):
        T = mrrxor.transition_tensor(0.0)
        beliefs = mrrxor.beliefs(T, mrrxor.sample(T, 500, 32, torch.Generator().manual_seed(8)))[:, :-1]
        s = probe.shuffled(beliefs, seed=0)
        rows, shuffled_rows = beliefs.flatten(0, 1).round(decimals=10), s.flatten(0, 1)
        values = torch.unique(rows, dim=0)
        self.assertTrue(torch.equal(torch.unique(shuffled_rows, dim=0), values))  # the same set of beliefs
        mapping = {}
        for b, t in zip(map(tuple, rows.tolist()), map(tuple, shuffled_rows.tolist())):
            self.assertEqual(mapping.setdefault(b, t), t)  # equal beliefs get equal targets
        self.assertEqual(len(set(mapping.values())), len(mapping))  # a bijection
        self.assertTrue(any(b != t for b, t in mapping.items()))

    def test_probe_alignment_end_to_end(self):
        """With features equal to the current belief, the probe recovers the belief and the next-token distribution
        exactly, while the shuffled control does not. A one-position lag must lower R^2 clearly."""
        T = mrrxor.transition_tensor(0.05)
        exact = probe.probe(BeliefModel(T, lag=0), T, count=2048)
        for weighting in ("uniform", "process"):
            self.assertAlmostEqual(exact[f"r2_{weighting}"], 1.0, places=5)
            self.assertAlmostEqual(exact[f"next_r2_{weighting}"], 1.0, places=5)
            self.assertLess(exact[f"shuffled_r2_{weighting}"], 0.5)
            self.assertIsNone(exact[f"r2_by_position_{weighting}"][0])
            self.assertTrue(all(r > 1 - 1e-5 for r in exact[f"r2_by_position_{weighting}"][1:]))
            for split in ("parity_", "early_", "late_"):  # no history is shared between fitting and scoring
                self.assertGreater(exact[f"{split}r2_{weighting}"], 0.999, split)
                self.assertGreater(exact[f"{split}next_r2_{weighting}"], 0.999, split)
                self.assertLess(exact[f"{split}shuffled_r2_{weighting}"], 0.1, split)
        lagged = probe.probe(BeliefModel(T, lag=1), T, count=2048)
        self.assertLess(lagged["r2_uniform"], 0.9)

    def test_held_out_positions_fit_and_score_the_stated_positions(self):
        """Without belief information at the fitting or the scoring positions of a held-out-position split, its probe
        cannot recover the belief, while the random-split probe still can from the other positions."""
        T = mrrxor.transition_tensor(0.05)
        for blank, splits in (("odd", ["parity_"]), ("even", ["parity_"]), ("first", ["early_", "late_"]),
                              ("second", ["early_", "late_"])):
            result = probe.probe(BeliefModel(T, blank=blank), T, count=2048)
            self.assertGreater(result["r2_uniform"], 0.3, blank)
            for split in splits:
                self.assertLess(result[f"{split}r2_uniform"], 0.1, (blank, split))

    def test_scripts_end_to_end(self):
        """train.py, probe.py (final and all checkpoints) and summarize.py on a tiny model for 30 steps."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "runs" / "L1_d16" / "eps0.05" / "s0"
            python = [sys.executable]
            subprocess.run(python + [str(ROOT / "train.py"), "--layers", "1", "--d-model", "16", "--steps", "30",
                                     "--warmup", "5", "--log-every", "10", "--ckpt-every", "10", "--out", str(run)],
                           check=True, capture_output=True, cwd=ROOT)
            self.assertEqual(sorted(p.name for p in run.parent.iterdir()), ["s0", "s0.pt"])
            self.assertEqual(sorted(p.name for p in run.iterdir()),
                             ["step_000010.pt", "step_000020.pt", "step_000030.pt"])
            final = torch.load(f"{run}.pt", weights_only=True)
            self.assertEqual(final["step"], 30)
            self.assertTrue(np.isfinite(final["test_kl"]))
            self.assertEqual({v.device.type for v in final["model"].values()}, {"cpu"})
            for name, checkpoint in ((f"{run}.pt", "final"), (str(run), "all")):  # a run with or without .pt
                subprocess.run(python + [str(ROOT / "probe.py"), name, "--trajectories", "256", "--checkpoint",
                                         checkpoint], check=True, capture_output=True, cwd=ROOT)
            result = json.loads(Path(f"{run}.json").read_text())
            self.assertEqual(result["layers"], 1)
            for key in ("r2", "next_r2", "shuffled_r2", "init_r2", "parity_r2", "parity_next_r2",
                        "parity_shuffled_r2", "init_parity_r2"):
                for weighting in ("uniform", "process"):
                    self.assertTrue(np.isfinite(result[f"{key}_{weighting}"]), key)
            self.assertEqual(len(list(run.glob("probe_step_*.json"))), 3)
            summary = subprocess.run(python + [str(ROOT / "summarize.py")], check=True, capture_output=True,
                                     text=True, cwd=tmp).stdout
            self.assertIn("### ε = 0.05", summary)
            self.assertIn("**Test KL and belief-state R² by probe split**", summary)
            self.assertIn("| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |", summary)
            self.assertIn(f"| 1 layers, d_model 16 | {final['test_kl']:.1e} | ".replace("e-0", "e-"), summary)
            self.assertIn("| model | next-token distribution | shuffled belief state | untrained network |", summary)
            self.assertIn("<summary>Process weighting</summary>", summary)
            self.assertIn("Seeds per row: 1.", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
