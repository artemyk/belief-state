"""Self-contained checks of the process, the training targets, the model, and the probe.

Run from the repository root with `python tests.py` (about half a minute). The process and its Bayesian
quantities are compared with a brute-force reference that enumerates hidden-state paths and is written independently
of mrrxor.py. The model is checked for causality and for the features it returns. The probe is checked for the
alignment of activations with beliefs, all four splits, its weighting, its ridge fit, and its controls. End-to-end
checks run train.py, probe.py and summarize.py on a tiny model and cover provenance, caching, and undefined scores.
"""
import copy
import hashlib
import itertools
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

import mrrxor
import probe
import probing
import summarize
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
    def test_mixing_probability_bounds(self):
        for eps in (-0.1, 1.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(eps=eps), self.assertRaisesRegex(ValueError, "eps"):
                mrrxor.transition_tensor(eps)
        for eps in (0.0, 1.0):
            T = mrrxor.transition_tensor(eps)
            self.assertTrue((T >= 0).all())
            torch.testing.assert_close(T.sum((0, 2)), torch.ones(5, dtype=T.dtype))

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
    def test_invalid_float_settings_are_rejected_before_initialization(self):
        import train
        for option in ("--lr", "--init-std"):
            for value in ("nan", "inf", "-inf", "-0.1"):
                with self.subTest(option=option, value=value), tempfile.TemporaryDirectory() as directory:
                    run = Path(directory)/"s0"
                    stderr = io.StringIO()
                    with patch.object(sys, "argv", ["train.py", f"{option}={value}", "--out", str(run)]), \
                         patch.object(sys, "stderr", stderr), \
                         patch.object(train, "untrained", side_effect=AssertionError("Model creation must not start")):
                        with self.assertRaises(SystemExit) as error:
                            train.main()
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn(f"{option} must be finite and nonnegative", stderr.getvalue())
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_zero_learning_rate_and_initialization_scale_remain_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/"s0"
            subprocess.run([sys.executable, str(ROOT/"train.py"), "--layers", "1", "--d-model", "8",
                            "--context", "8", "--batch", "2", "--steps", "1", "--warmup", "0",
                            "--lr", "0", "--init-std", "0", "--out", str(run)],
                           check=True, cwd=ROOT, capture_output=True)
            saved = torch.load(Path(f"{run}.pt"), weights_only=True)
            expected = untrained(Config(**saved["config"]), 0, saved["args"]["seed"])
            self.assertEqual(saved["step"], 1)
            self.assertTrue(np.isfinite(saved["test_kl"]))
            for name, value in expected.state_dict().items():
                torch.testing.assert_close(saved["model"][name], value, atol=0, rtol=0)

    def test_nonfinite_training_values_prevent_updates_and_checkpoint_writes(self):
        import train
        # Load optimizer machinery before mocking hardware queries and loss.
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))])
        original_model, original_loss = train.untrained, train.F.cross_entropy
        tokens = torch.zeros(2, 8, dtype=torch.long)
        optimal = torch.full((2, 8, 2), .5, dtype=torch.float64)
        for mode in ("loss", "gradient", "state"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)/"s0"
                def model_factory(*args, **kwargs):
                    model = original_model(*args, **kwargs)
                    if mode == "gradient":
                        model.embed.weight.register_hook(lambda g: torch.full_like(g, float("inf")))
                    elif mode == "state":
                        # The held-out inputs below do not read token 1, so KL
                        # alone cannot detect this invalid embedding row.
                        with torch.no_grad():
                            model.embed.weight[1].fill_(float("nan"))
                    return model
                def loss(logits, targets):
                    value = original_loss(logits, targets)
                    return value*float("nan") if mode == "loss" else value
                def unexpected_update(*args, **kwargs):
                    self.fail("Bad values reached the optimizer")
                argv = ["train.py", "--layers", "1", "--d-model", "8", "--context", "8", "--batch", "2",
                        "--steps", "0" if mode == "state" else "1", "--warmup", "0", "--out", str(run)]
                expected_error = RuntimeError if mode == "gradient" else FloatingPointError
                message = {"loss": "Nonfinite training loss", "gradient": "non-finite", "state": "Nonfinite model state"}[mode]
                with patch.object(sys, "argv", argv), \
                     patch.object(torch.backends.mps, "is_available", return_value=False), \
                     patch.object(torch.cuda, "is_available", return_value=False), \
                     patch.object(train, "untrained", side_effect=model_factory), \
                     patch.object(train.F, "cross_entropy", side_effect=loss), \
                     patch.object(train.mrrxor, "sample", return_value=tokens), \
                     patch.object(train.mrrxor, "test_set", return_value=(tokens, optimal)), \
                     patch.object(torch.optim.Adam, "step", new=unexpected_update):
                    with self.assertRaisesRegex(expected_error, message):
                        train.main()
                self.assertEqual(list(Path(directory).rglob("*.pt")), [])

    def test_divergent_finite_learning_rate_never_saves_a_checkpoint(self):
        import train
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))])
        for interval in (1, 1000):
            with self.subTest(checkpoint_interval=interval), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)/"s0"
                argv = ["train.py", "--layers", "1", "--d-model", "8", "--context", "8", "--batch", "2",
                        "--steps", "1", "--warmup", "0", "--lr", "1e20", "--log-every", "1000",
                        "--ckpt-every", str(interval), "--out", str(run)]
                stdout = io.StringIO()
                with patch.object(sys, "argv", argv), patch.object(sys, "stdout", stdout), \
                     patch.object(torch.backends.mps, "is_available", return_value=False), \
                     patch.object(torch.cuda, "is_available", return_value=False):
                    with self.assertRaisesRegex(FloatingPointError, "Nonfinite test KL"):
                        train.main()
                self.assertEqual(list(Path(directory).rglob("*.pt")), [])
                self.assertNotIn("trained for", stdout.getvalue())

    def test_invalid_counts_are_rejected_before_initialization_or_output_creation(self):
        import train
        cases = [("--steps", "-1"), ("--warmup", "-1"),
                 ("--log-every", "0"), ("--log-every", "-1"),
                 ("--ckpt-every", "0"), ("--ckpt-every", "-1"),
                 ("--batch", "0"), ("--context", "0"),
                 ("--layers", "0"), ("--d-model", "0")]
        for option, value in cases:
            with self.subTest(option=option, value=value), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)/"s0"
                stderr = io.StringIO()
                with patch.object(sys, "argv", ["train.py", option, value, "--out", str(run)]), \
                     patch.object(sys, "stderr", stderr), \
                     patch.object(train, "untrained", side_effect=AssertionError("Model creation must not start")):
                    with self.assertRaises(SystemExit) as error:
                        train.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn(option, stderr.getvalue())
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_zero_steps_and_warmup_save_the_untrained_model(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/"s0"
            subprocess.run([sys.executable, str(ROOT/"train.py"), "--layers", "1", "--d-model", "8",
                            "--context", "8", "--steps", "0", "--warmup", "0", "--out", str(run)],
                           check=True, cwd=ROOT, capture_output=True)
            saved = torch.load(Path(f"{run}.pt"), weights_only=True)
            expected = untrained(Config(**saved["config"]), saved["args"]["init_std"], saved["args"]["seed"])
            self.assertEqual(saved["step"], 0)
            self.assertTrue(np.isfinite(saved["test_kl"]))
            self.assertEqual(list(run.iterdir()), [])
            for name, value in expected.state_dict().items():
                torch.testing.assert_close(saved["model"][name], value, atol=0, rtol=0)

    def test_sweep_stops_and_reaps_training_jobs_on_failure_and_signals(self):
        shells = ["sh"]
        if bash := shutil.which("bash"):
            shells.append(bash)
        for shell in shells:
            for mode in ("failure", "interrupt", "terminate"):
                with self.subTest(shell=shell, mode=mode), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    python = root/"fake-python"
                    python.write_text(f"#!{sys.executable}\n" + '''import os, signal, sys, time
from pathlib import Path
root = Path(os.environ["SWEEP_TEST_ROOT"])
if sys.argv[1] != "train.py":
    (root/"unexpected-probe-or-summary").touch()
    sys.exit(1)
eps = sys.argv[sys.argv.index("--eps")+1]
if os.environ["SWEEP_TEST_MODE"] == "failure" and eps == "0.05":
    deadline = time.monotonic()+5
    while not (root/"ready-0").exists() and time.monotonic() < deadline:
        time.sleep(.01)
    sys.exit(7)
def stopped(signum, frame):
    (root/("stopped-"+eps)).write_text(str(signum))
    sys.exit(0)
signal.signal(signal.SIGTERM, stopped)
(root/("ready-"+eps)).write_text(str(os.getpid()))
while True:
    time.sleep(.1)
''')
                    python.chmod(0o755)
                    env = dict(os.environ, PYTHON=str(python), SEEDS="0", RUNS_DIR=str(root/"runs"),
                               SWEEP_TEST_ROOT=str(root), SWEEP_TEST_MODE=mode)
                    process = subprocess.Popen([shell, str(ROOT/"run_sweep.sh")], cwd=ROOT, env=env,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    try:
                        if mode != "failure":
                            deadline = time.monotonic()+5
                            while len(list(root.glob("ready-*"))) < 2 and time.monotonic() < deadline:
                                time.sleep(.01)
                            self.assertEqual(len(list(root.glob("ready-*"))), 2)
                            process.send_signal(signal.SIGINT if mode == "interrupt" else signal.SIGTERM)
                        stdout, stderr = process.communicate(timeout=10)
                        self.assertEqual(process.returncode, {"failure": 7, "interrupt": 130, "terminate": 143}[mode], stderr)
                        expected = {"0"} if mode == "failure" else {"0", "0.05"}
                        for eps in expected:
                            self.assertTrue((root/f"stopped-{eps}").exists(), stderr)
                            pid = int((root/f"ready-{eps}").read_text())
                            with self.assertRaises(ProcessLookupError):
                                os.kill(pid, 0)
                        self.assertFalse((root/"unexpected-probe-or-summary").exists())
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.communicate()
                        # A failing regression must not leave its fake trainers running.
                        for ready in root.glob("ready-*"):
                            if not (root/ready.name.replace("ready-", "stopped-")).exists():
                                try:
                                    os.kill(int(ready.read_text()), signal.SIGTERM)
                                except ProcessLookupError:
                                    pass

    def test_existing_run_artifacts_are_rejected_before_model_creation(self):
        import train
        for artifact in ("directory", "checkpoint", "probe"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = root/"s0"
                if artifact == "directory":
                    run.mkdir()
                    (run/"step_100000.pt").write_bytes(b"old checkpoint")
                else:
                    suffix = "pt" if artifact == "checkpoint" else "json"
                    Path(f"{run}.{suffix}").write_bytes(b"old result")
                before = {p.relative_to(root): p.read_bytes() if p.is_file() else None for p in root.rglob("*")}
                stderr = io.StringIO()
                with patch.object(sys, "argv", ["train.py", "--out", str(run)]), \
                     patch.object(sys, "stderr", stderr), \
                     patch.object(train, "untrained", side_effect=AssertionError("Model creation must not start")):
                    with self.assertRaises(SystemExit) as error:
                        train.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn("fresh run directory", stderr.getvalue())
                self.assertEqual(before, {p.relative_to(root): p.read_bytes() if p.is_file() else None
                                          for p in root.rglob("*")})

    def test_invalid_mixing_probability_does_not_create_output(self):
        import train
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/"s0"
            with patch.object(sys, "argv", ["train.py", "--eps", "-0.1", "--out", str(run)]), \
                 patch.object(sys, "stderr", io.StringIO()), \
                 patch.object(train, "untrained", side_effect=AssertionError("Model creation must not start")):
                with self.assertRaises(SystemExit) as error:
                    train.main()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_sweep_uses_a_fresh_output_root_and_refuses_to_repeat_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root/"new runs"
            calls = root/"calls.jsonl"
            python = root/"fake python"
            python.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
with Path(os.environ["SWEEP_TEST_LOG"]).open("a") as log:
    log.write(json.dumps(sys.argv[1:])+"\\n")
if sys.argv[1] == "train.py":
    run = Path(sys.argv[sys.argv.index("--out")+1])
    run.mkdir()
    Path(str(run)+".pt").write_bytes(b"new checkpoint")
''')
            python.chmod(0o755)
            environment = dict(os.environ, PYTHON=str(python), SEEDS="0", RUNS_DIR=str(runs), SWEEP_TEST_LOG=str(calls))
            command = ["sh", str(ROOT/"run_sweep.sh")]
            subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True)
            recorded = [json.loads(line) for line in calls.read_text().splitlines()]
            training = [args for args in recorded if args[0] == "train.py"]
            self.assertEqual(len(training), 2)
            self.assertEqual({Path(args[args.index("--out")+1]) for args in training},
                             {runs/f"L4_d64_rope/eps{eps}/s0" for eps in ("0", "0.05")})
            self.assertEqual(recorded[-1], ["summarize.py", str(runs)])
            probing_call = next(args for args in recorded if args[0] == "probe.py")
            self.assertEqual(set(probing_call[1:]), {str(p) for p in runs.rglob("*.pt")})
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            refused = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("fresh RUNS_DIR", refused.stderr)
            self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()})

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
        self.model = untrained(Config(context=16, n_layers=2, d_model=32, d_mlp=128, rope=False), 0.1, seed=0).eval()

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

    def test_bos_visible_on_constant_histories(self):
        """RoPE can distinguish one 1 from five 1s, and the BOS embedding affects both predictions."""
        model = untrained(Config(rope=True), 0.1, seed=0).eval()
        tokens = torch.ones(1, 6, dtype=torch.long)
        with torch.no_grad():
            logits = model(tokens)
            self.assertGreater(float((logits[:, 1] - logits[:, 5]).abs().max()), 1e-4)
            model.embed.weight[model.cfg.vocab].zero_()
            difference = (logits - model(tokens)).abs().amax(-1)
            self.assertTrue((difference[:, [1, 5]] > 1e-4).all())

    def test_probe_loader_preserves_standard_bos_and_rejects_other_metadata(self):
        tokens = torch.tensor([[1, 0, 1, 1]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"teacher.pt"
            config = vars(self.model.cfg).copy()
            for explicit_metadata in (False, True):
                if explicit_metadata:
                    config["bos_visible"] = True
                torch.save({"config": config, "model": self.model.state_dict()}, path)
                restored, _ = probe.load(path)
                torch.testing.assert_close(restored(tokens), self.model(tokens))
            config["bos_visible"] = False
            torch.save({"config": config, "model": self.model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "standard BOS"):
                probe.load(path)

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
                x, _, _ = block(x)
            torch.testing.assert_close(features[..., 64:], m.ln_final(x))
            torch.testing.assert_close(m.unembed(features[..., 64:]), logits)

    def test_cached_block_matches_full_execution_and_preserves_cache_gradients(self):
        """One block implementation serves full sequences and arbitrary written KV caches."""
        for rope in (False, True):
            model = untrained(Config(context=16, n_layers=1, d_model=16, d_mlp=32, rope=rope), .1, seed=11).double()
            block = model.blocks[0]
            x = torch.randn(2, 9, 16, dtype=torch.float64)
            full, _, full_cache = block(x)
            _, _, prefix = block(x[:, :5])
            actual, _, extended = block(x[:, 5:], cache=prefix)
            torch.testing.assert_close(actual, full[:, 5:], atol=1e-10, rtol=1e-10)
            for a, b in zip(extended, full_cache):
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
            written = tuple(value.detach().requires_grad_(True) for value in prefix)
            result, _, _ = block(x[:, 5:], cache=written)
            gradients = torch.autograd.grad(result.square().sum(), written)
            self.assertTrue(all(gradient.norm() > 0 for gradient in gradients))

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
        self.assertEqual(count(Config(rope=False)), 144_418)
        self.assertEqual(count(Config()), 142_370)
        self.assertEqual(count(Config(n_layers=2, d_model=32, d_mlp=128, rope=False)), 20_370)
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
    def test_checkpoint_selection_orders_step_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            steps = [1000000, 5000, 1000001, 995000]
            for step in steps:
                (run/f"step_{step:06d}.pt").touch()
            expected = [f"step_{step:06d}" for step in sorted(steps)]
            self.assertEqual(probe.checkpoint_names(run, "all"), expected)
            self.assertEqual(probe.checkpoint_names(run, "latest"), expected[-1:])
            self.assertEqual(probe.checkpoint_names(run, "step_005000"), ["step_005000"])

    def test_test_fraction_bounds(self):
        T = mrrxor.transition_tensor(0.05)
        for fraction in (-0.2, 0.0, 1.0, 1.2, float("nan"), float("inf"), -float("inf")):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "test_fraction"):
                    probing.split(100, fraction, seed=0)
                with patch.object(probing, "trajectories", side_effect=AssertionError("Sampling must not start")):
                    with self.assertRaisesRegex(ValueError, "test_fraction"):
                        probing.prepare_data(T, 32, count=100, test_fraction=fraction)

    def test_split_is_by_trajectory(self):
        test = probing.split(1000, 0.2, seed=0)
        self.assertEqual(int(test.sum()), 200)
        self.assertTrue(torch.equal(test, probing.split(1000, 0.2, seed=0)))
        self.assertFalse(torch.equal(test, probing.split(1000, 0.2, seed=1)))

    def test_cached_probe_requires_matching_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s0.pt"
            path.write_bytes(b"checkpoint contents")
            checkpoint = {"config": vars(Config()), "args": {"seed": "0", "eps": "0.05", "context": "32"}}
            settings = dict(count=128, test_fraction=0.2, seed=0)
            metadata = probe.result_metadata(path, checkpoint, settings)
            self.assertEqual(metadata["checkpoint_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(metadata["training_settings"], {"seed": 0, "eps": 0.05})
            self.assertEqual(metadata["config"]["context"], 32)
            cached = path.with_suffix(".json")
            self.assertFalse(probe.result_is_current(cached, metadata))
            cached.write_text(json.dumps(metadata))
            self.assertTrue(probe.result_is_current(cached, metadata))
            changed = probe.result_metadata(path, checkpoint, {**settings, "seed": 1})
            self.assertFalse(probe.result_is_current(cached, changed))
            changed = copy.deepcopy(metadata)
            changed["probe_settings"]["code_sha256"] = "changed implementation"
            self.assertFalse(probe.result_is_current(cached, changed))
            path.write_bytes(b"replacement checkpoint")
            self.assertFalse(probe.result_is_current(cached, probe.result_metadata(path, checkpoint, settings)))
            cached.write_text("{")
            self.assertFalse(probe.result_is_current(cached, metadata))

    def test_ridge_cv_against_explicit_fold_predictions(self):
        """Verify position-MSE selection and the full refit against independent weighted ridge solves."""
        gen = torch.Generator().manual_seed(7)
        x = torch.randn(60, 8, 8, generator=gen, dtype=torch.float64)
        y = torch.randn(60, 8, 3, generator=gen, dtype=x.dtype)
        y[:, :, 0] += 5 + 3 * x[:, :, 0]
        y[:, 0, 0] = 3  # a constant-target position still contributes validation squared error
        weights = 0.1 + torch.rand(60, generator=gen, dtype=x.dtype)

        def solve(a, b, w, alpha):
            w = w / w.sum()
            xm, ym = w @ a, w @ b
            ac = a - xm
            cov = ac.T @ (w[:, None] * ac)
            slope = torch.linalg.solve(cov + alpha * cov.diagonal().mean() * torch.eye(a.shape[1], dtype=x.dtype),
                                       ac.T @ (w[:, None] * (b - ym)))
            return torch.cat(((ym - xm @ slope)[None, :], slope))

        scores = torch.zeros(len(probing.ALPHAS), 2, dtype=x.dtype)
        for position in range(x.shape[1]):
            train = torch.arange(x.shape[1]) != position
            tx, ty = x[:, train].flatten(0, 1), y[:, train].flatten(0, 1)
            tw = weights.repeat_interleave(int(train.sum()))
            for k, alpha in enumerate(probing.ALPHAS):
                coef = solve(tx, ty, tw, alpha)
                pred = x[:, position] @ coef[1:] + coef[0]
                for j, error in enumerate((y[:, position] - pred).split([1, 2], 1)):
                    scores[k, j] += weights @ error.square().sum(1) / weights.sum()
        scores /= x.shape[1]
        coef, alphas = probing.fit_ridge_cv(probing.position_grams(x, y, weights), x.shape[-1], [1, 2])
        self.assertEqual(alphas, probing.ALPHAS[scores.argmin(0)].tolist())
        all_weights = weights.repeat_interleave(x.shape[1])
        fits = [solve(x.flatten(0, 1), y.flatten(0, 1), all_weights, alpha) for alpha in alphas]
        torch.testing.assert_close(coef, torch.cat((fits[0][:, :1], fits[1][:, 1:]), 1))

    def test_probe_validation_uses_only_fitting_rows(self):
        """All splits select alpha using only their fitting lengths and outer training trajectories."""
        T = mrrxor.transition_tensor(0.05)
        data = probing.prepare_data(T, 32, count=128)
        with patch.object(probing, "position_grams", wraps=probing.position_grams) as statistics, \
                patch.object(probing, "fit_ridge_cv", wraps=probing.fit_ridge_cv) as fits:
            probing.probe(BeliefModel(T), data)
        self.assertEqual(statistics.call_count, 2)  # one set of statistics per weighting, shared by all four splits
        self.assertEqual(len(fits.call_args_list), 8)
        train = ~probing.split(128, 0.2, seed=0)
        expected_lengths = [list(range(1, 32)), list(range(1, 32, 2)), list(range(1, 16)), list(range(16, 32))]
        for weighting, gram_call, calls in zip(("uniform", "process"), statistics.call_args_list,
                                              (fits.call_args_list[:4], fits.call_args_list[4:])):
            tokens, weights = probing.trajectories(T, 32, weighting, 128, seed=0)
            beliefs = mrrxor.beliefs(T, tokens)
            x, y, w = gram_call.args
            self.assertEqual(x.shape[:2], (102, 31))
            torch.testing.assert_close(y[:, :, :5], beliefs[train][:, 1:-1])
            torch.testing.assert_close(w, weights[train].double())
            full_grams = probing.position_grams(x, y, w)
            for call, lengths in zip(calls, expected_lengths):
                grams, n_features, sizes = call.args
                torch.testing.assert_close(grams, full_grams[torch.tensor(lengths) - 1])
                self.assertEqual(n_features, x.shape[-1])
                self.assertEqual(sizes, [5, 2, 5])

    def test_evaluation_shares_data_between_networks(self):
        model = untrained(Config(context=8, n_layers=1, d_model=8, d_mlp=32), 0.1, seed=0).eval()
        with patch.object(probing, "prepare_data", wraps=probing.prepare_data) as prepare, \
                patch.object(probing, "probe", wraps=probing.probe) as fits:
            probing.evaluate(model, {"eps": 0.05, "init_std": 0.1, "seed": 0},
                             {"count": 128, "test_fraction": 0.2, "seed": 0}, include_initial=True)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(fits.call_count, 2)
        self.assertIs(fits.call_args_list[0].args[1], fits.call_args_list[1].args[1])

    def test_uniform_weighting(self):
        """eps > 0: uniformly random strings. eps = 0: process samples weighted by 1 / probability, which makes every
        possible sequence equally likely."""
        tokens, weights = probing.trajectories(mrrxor.transition_tensor(0.05), 32, "uniform", 1000, seed=0)
        self.assertTrue((weights == 1).all() and set(tokens.unique().tolist()) == {0, 1})
        T = mrrxor.transition_tensor(0.0)
        tokens, weights = probing.trajectories(T, 6, "uniform", 200_000, seed=0)
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
        s = probing.shuffled(beliefs, seed=0)
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
        data = probing.prepare_data(T, 32, count=2048)
        exact = probing.probe(BeliefModel(T, lag=0), data)
        for weighting in ("uniform", "process"):
            self.assertAlmostEqual(exact[f"r2_{weighting}"], 1.0, places=5)
            self.assertAlmostEqual(exact[f"next_r2_{weighting}"], 1.0, places=5)
            self.assertLess(exact[f"shuffled_r2_{weighting}"], 0.5)
            self.assertIsNone(exact[f"r2_by_position_{weighting}"][0])
            self.assertIsNone(exact[f"next_r2_by_position_{weighting}"][1])
            self.assertTrue(all(r > 1 - 1e-5 for r in exact[f"r2_by_position_{weighting}"][1:]))
            for split in ("parity_", "early_", "late_"):  # no history is shared between fitting and scoring
                self.assertGreater(exact[f"{split}r2_{weighting}"], 0.999, split)
                self.assertGreater(exact[f"{split}next_r2_{weighting}"], 0.999, split)
                self.assertLess(exact[f"{split}shuffled_r2_{weighting}"], 0.1, split)
        lagged = probing.probe(BeliefModel(T, lag=1), data)
        self.assertLess(lagged["r2_uniform"], 0.9)

    def test_held_out_positions_fit_and_score_the_stated_positions(self):
        """Without belief information at the fitting or the scoring positions of a held-out-position split, its probe
        cannot recover the belief, while the random-split probe still can from the other positions."""
        T = mrrxor.transition_tensor(0.05)
        data = probing.prepare_data(T, 32, count=2048)
        for blank, splits in (("odd", ["parity_"]), ("even", ["parity_"]), ("first", ["early_", "late_"]),
                              ("second", ["early_", "late_"])):
            result = probing.probe(BeliefModel(T, blank=blank), data)
            self.assertGreater(result["r2_uniform"], 0.3, blank)
            for split in splits:
                self.assertLess(result[f"{split}r2_uniform"], 0.1, (blank, split))

    def test_scripts_end_to_end(self):
        """train.py, probe.py (final and all checkpoints) and summarize.py on a tiny model for 30 steps."""
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "runs" / "L1_d16_rope" / "eps0.05" / "s0"
            python = [sys.executable]
            subprocess.run(python + [str(ROOT / "train.py"), "--layers", "1", "--d-model", "16", "--steps", "30",
                                     "--warmup", "5", "--log-every", "10", "--ckpt-every", "10", "--out", str(run)],
                           check=True, capture_output=True, cwd=ROOT)
            self.assertEqual(sorted(p.name for p in run.parent.iterdir()), ["s0", "s0.pt"])
            self.assertEqual(sorted(p.name for p in run.iterdir()),
                             ["step_000010.pt", "step_000020.pt", "step_000030.pt"])
            final = torch.load(f"{run}.pt", weights_only=True)
            self.assertEqual(final["step"], 30)
            self.assertEqual(final["config"], vars(Config(n_layers=1, d_model=16, d_mlp=64)))
            self.assertEqual(final["args"]["seed"], 0)
            self.assertEqual(final["args"]["eps"], 0.05)
            self.assertTrue(np.isfinite(final["test_kl"]))
            self.assertEqual({v.device.type for v in final["model"].values()}, {"cpu"})
            before = {p: p.read_bytes() for p in Path(tmp).rglob("*") if p.is_file()}
            repeated = subprocess.run(python + [str(ROOT/"train.py"), "--steps", "1", "--out", str(run)],
                                      capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("fresh run directory", repeated.stderr)
            self.assertEqual(before, {p: p.read_bytes() for p in Path(tmp).rglob("*") if p.is_file()})
            for name, checkpoint in ((f"{run}.pt", "final"), (str(run), "all")):  # a run with or without .pt
                subprocess.run(python + [str(ROOT / "probe.py"), name, "--trajectories", "256", "--checkpoint",
                                         checkpoint], check=True, capture_output=True, cwd=ROOT)
            result = json.loads(Path(f"{run}.json").read_text())
            self.assertEqual(result["config"]["n_layers"], 1)
            self.assertEqual(result["step"], 30)
            for key in ("r2", "next_r2", "shuffled_r2", "init_r2", "parity_r2", "parity_next_r2",
                        "parity_shuffled_r2", "init_parity_r2"):
                for weighting in ("uniform", "process"):
                    self.assertTrue(np.isfinite(result[f"{key}_{weighting}"]), key)
            self.assertEqual(len(list(run.glob("probe_step_*.json"))), 3)
            cached = subprocess.run(python + [str(ROOT / "probe.py"), str(run), "--trajectories", "256",
                                             "--checkpoint", "all"], check=True, capture_output=True, text=True)
            self.assertEqual(cached.stdout, "")
            stale_path = run / "probe_step_000010.json"
            stale = json.loads(stale_path.read_text())
            stale["probe_settings"]["code_sha256"] = "old implementation"
            stale_path.write_text(json.dumps(stale))
            refreshed = subprocess.run(python + [str(ROOT / "probe.py"), str(run), "--trajectories", "256",
                                                "--checkpoint", "all"], check=True, capture_output=True, text=True)
            self.assertEqual(len(refreshed.stdout.splitlines()), 1)
            self.assertIn("step_000010", refreshed.stdout)
            self.assertEqual(json.loads(stale_path.read_text())["probe_settings"], result["probe_settings"])
            summary = subprocess.run(python + [str(ROOT / "summarize.py")], check=True, capture_output=True,
                                     text=True, cwd=tmp).stdout
            self.assertIn("### ε = 0.05", summary)
            self.assertIn("**Test KL and belief-state R² by probe split**", summary)
            self.assertIn("| model | test KL | random | odd/even | 1st → 2nd half | 2nd → 1st half |", summary)
            self.assertIn(f"| 1 layers, d_model 16, RoPE | {final['test_kl']:.1e} | ".replace("e-0", "e-"), summary)
            self.assertIn("| model | next-token distribution | shuffled belief state | untrained network |", summary)
            self.assertIn("<summary>Process weighting</summary>", summary)
            self.assertIn("Seeds per row: 1.", summary)

    def test_constant_target_reporting(self):
        """At eps = 0.8 the next-bit distribution is constant, so the CLI and tables must handle undefined R²."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(context=8, n_layers=1, d_model=8, d_mlp=32)
            model = untrained(cfg, 0.1, seed=0)
            run = Path(tmp) / "runs/L1_d8_rope/eps0.8/s0"
            run.parent.mkdir(parents=True)
            torch.save({"config": vars(cfg), "args": {"eps": "0.8", "seed": "0", "init_std": "0.1"},
                        "step": 0, "model": model.state_dict()}, f"{run}.pt")
            completed = subprocess.run([sys.executable, str(ROOT / "probe.py"), str(run), "--trajectories", "128"],
                                       check=True, capture_output=True, text=True)
            self.assertIn("next-token undefined, undefined", completed.stdout)
            result = json.loads(Path(f"{run}.json").read_text())
            self.assertIsNone(result["next_r2_uniform"])
            self.assertIsNone(result["next_r2_process"])
            summary = subprocess.run([sys.executable, str(ROOT / "summarize.py")], cwd=tmp,
                                     check=True, capture_output=True, text=True)
            self.assertIn("undefined", summary.stdout)


class Summary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paths = []
        for seed in (0, 1):
            path = self.root / f"runs/L1_d8_rope/eps0.05/s{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Only the file identity matters here; the reporting script does not load model weights.
            path.write_bytes(f"checkpoint {seed}".encode())
            checkpoint = {"config": vars(Config(n_layers=1, d_model=8, d_mlp=32)),
                          "args": {"eps": "0.05", "seed": str(seed), "lr": "0.001"}}
            result = {"step": 30, "test_kl": 0.01,
                      **probe.result_metadata(path, checkpoint, dict(count=128, test_fraction=0.2, seed=0))}
            for split in ("", "parity_", "early_", "late_"):
                for prefix, target in (("", ""), ("", "next_"), ("", "shuffled_"), ("init_", "")):
                    for weighting in ("uniform", "process"):
                        result[f"{prefix}{split}{target}r2_{weighting}"] = 0.5 + 0.1 * seed
            path = path.with_suffix(".json")
            path.write_text(json.dumps(result))
            self.paths.append(path)

    def test_summary_rejects_nonstandard_bos_metadata(self):
        path = self.paths[0]
        result = json.loads(path.read_text())
        result["config"]["bos_visible"] = True
        path.write_text(json.dumps(result))
        self.assertEqual(sum(len(rows) for rows in summarize.load_runs(self.paths).values()), 2)
        result["config"]["bos_visible"] = False
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "standard BOS"):
            summarize.load_runs(self.paths)

    def test_rejects_incompatible_results(self):
        path = self.paths[1]
        original = json.loads(path.read_text())
        cases = [(("probe_settings", "seed"), 1, "incompatible probe settings"),
                 (("config", "context"), 64, "incompatible model or training settings"),
                 (("training_settings", "lr"), 0.1, "incompatible model or training settings"),
                 (("step",), 100, "incompatible model or training settings"),
                 (("checkpoint_sha256",), "wrong hash", "hash mismatch")]
        for keys, value, message in cases:
            with self.subTest(keys=keys):
                result = copy.deepcopy(original)
                field = result
                for key in keys[:-1]:
                    field = field[key]
                field[keys[-1]] = value
                path.write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, message):
                    summarize.load_runs(self.paths)

    def test_rejects_duplicate_seeds_and_missing_provenance(self):
        path = self.paths[1]
        result = json.loads(path.read_text())
        result["training_settings"]["seed"] = 0
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "duplicate training seed"):
            summarize.load_runs(self.paths)
        for field in ("checkpoint_sha256", "step"):
            incomplete = {key: value for key, value in result.items() if key != field}
            path.write_text(json.dumps(incomplete))
            with self.assertRaisesRegex(ValueError, "missing provenance"):
                summarize.load_runs(self.paths)

    def test_prints_only_and_preserves_undefined_scores(self):
        result = json.loads(self.paths[0].read_text())
        result["r2_uniform"] = result["next_r2_uniform"] = None
        self.paths[0].write_text(json.dumps(result))
        (self.root / "README.md").write_text("Keep this document unchanged.\n")
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        markdown = summarize.render_summary(summarize.load_runs(self.paths))
        completed = subprocess.run([sys.executable, str(ROOT / "summarize.py")], cwd=self.root,
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, markdown + "\n")
        self.assertIn("undefined", completed.stdout)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_summary_accepts_an_explicit_runs_root(self):
        expected = summarize.render_summary(summarize.load_runs(self.paths))
        completed = subprocess.run([sys.executable, str(ROOT/"summarize.py"), str(self.root/"runs")],
                                   cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertEqual(completed.stdout, expected+"\n")

    def test_imports_do_not_execute_scripts(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); import train, probe, probing, summarize"
        completed = subprocess.run([sys.executable, "-c", code], cwd=self.root,
                                   check=True, capture_output=True, text=True)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main(verbosity=2)
