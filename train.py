"""Train the transformer on MRRXOR for a fixed number of steps.

The loss is the cross entropy of the model's prediction of each token of a sampled length-32 sequence, the first
included (the model reads a BOS token first), averaged over the 32 tokens. Progress is measured by the test KL: the KL
divergence from the Bayes-optimal next-token distribution (computed from the exact belief states) to the model's
prediction, averaged over all tokens of a fixed held-out set of 4,096 sequences. It is logged every `--log-every`
steps, and the current weights are saved to <out>/step_<step>.pt every `--ckpt-every` steps, to follow training.
<out>.pt holds the final weights and the training summary (steps, test KL).
The output directory, final checkpoint, and probe results must not already exist.

Usage: python train.py --eps 0.05 --seed 0 --out outputs/reproduction/L4_d64_rope/eps0.05/s0
"""
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import mrrxor
from model import Config, untrained

# Saved with every checkpoint: the training settings (the architecture is in "config").
RUN_SETTINGS = ("eps", "seed", "init_std", "batch", "lr", "warmup", "steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init-std", type=float, default=0.1,
                        help="std of the normal initialization of all weights")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64, help="residual-stream width; the MLPs have width 4 d_model")
    parser.add_argument("--rope", action=argparse.BooleanOptionalAction, default=True,
                        help="use rotary position embeddings (default); --no-rope uses learned absolute embeddings")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps", type=float, default=0.05, help="MRRXOR mixing probability")
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=100_000, help="number of steps (and of the cosine schedule)")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--out", type=Path, required=True, help="new run directory, e.g. outputs/reproduction/L4_d64_rope/eps0.05/s0 (no extension)")
    args = parser.parse_args()
    for name in ("layers", "d_model", "context", "batch", "log_every", "ckpt_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("steps", "warmup"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    for name in ("lr", "init_std"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    for path in (args.out, Path(f"{args.out}.pt"), Path(f"{args.out}.json")):
        if path.exists() or path.is_symlink():
            parser.error(f"Output already exists: {path}; choose a fresh run directory")
    try:
        T = mrrxor.transition_tensor(args.eps)
    except ValueError as error:
        parser.error(str(error))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    data_rng = torch.Generator().manual_seed(args.seed)

    cfg = Config(context=args.context, n_layers=args.layers, d_model=args.d_model, d_mlp=4 * args.d_model, rope=args.rope)
    model = untrained(cfg, args.init_std, args.seed).to(device)
    # Compiling merges the many small GPU kernels of this small model (its GPU time per step is about 1.5x shorter on
    # MPS), and on CUDA (mode reduce-overhead) also replays each step as a CUDA graph, about 1.25x faster on an L4; on the
    # CPU it is slower. The test KL and the saved weights use the uncompiled model, which shares its parameters.
    mode = "reduce-overhead" if device.type == "cuda" else "default"
    forward = model if device.type == "cpu" else torch.compile(model, mode=mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=True)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=args.warmup),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps - args.warmup),
                                                   eta_min=args.lr / 100),
    ], milestones=[args.warmup])

    args.out.mkdir(parents=True, exist_ok=False)
    test_tokens, test_optimal = mrrxor.test_set(T, args.context)

    def test_kl() -> float:
        with torch.no_grad():
            kl = mrrxor.kl_to_model(test_optimal, model(test_tokens.to(device)))
        if not math.isfinite(kl):
            raise FloatingPointError("Nonfinite test KL; training stopped")
        return kl

    def save(path: Path, step: int, **summary) -> None:
        state = {key: value.cpu() for key, value in model.state_dict().items()}
        if any(not bool(torch.isfinite(value).all()) for value in state.values()):
            raise FloatingPointError(f"Nonfinite model state at step {step}; checkpoint not saved")
        torch.save({"config": vars(cfg), "args": {k: getattr(args, k) for k in RUN_SETTINGS}, "step": step,
                    "model": state, **summary}, path)

    # Finite checks stop divergent updates. Validate predictions at logging and
    # checkpoint steps; checkpoint writes also check every model tensor.
    started = time.time()
    for step in range(1, args.steps + 1):
        tokens = mrrxor.sample(T, args.batch, args.context, data_rng).to(device)
        logits = forward(tokens)
        loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite training loss at step {step}; training stopped")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Check gradients without changing their magnitudes or the optimizer.
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"),
                                      norm_type=float("inf"), error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step % args.ckpt_every == 0:
            kl = test_kl()
        if step % args.log_every == 0:
            print(f"step {step:7d}  lr {scheduler.get_last_lr()[0]:.2e}  test KL {kl:.2e}"
                  f"  ({step / (time.time() - started):.0f} steps/s)", flush=True)
        if step % args.ckpt_every == 0:
            save(args.out / f"step_{step:06d}.pt", step)

    kl = test_kl()
    save(Path(f"{args.out}.pt"), args.steps, test_kl=kl, seconds=time.time() - started)
    print(f"trained for {args.steps} steps, test KL {kl:.2e}", flush=True)


if __name__ == "__main__":
    main()
