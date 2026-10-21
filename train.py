"""Train the transformer on MRRXOR until its next-token predictions are within a target KL of Bayes-optimal.

The loss is the usual next-token cross entropy on sampled tokens, averaged over all positions of each length-32
input. Progress is measured by the exact training KL: the KL divergence from the Bayes-optimal next-token
distribution (computed from the exact belief states) to the model's prediction, averaged over the positions and
sequences of the training batches. Training stops when its mean over the last `--window` steps falls below
`--kl-target`, or after `--max-steps`. The same KL on a fixed held-out set of 4,096 sequences (the test KL) is
logged every `--log-every` steps, and the current weights are saved to step_<step>.pt every `--ckpt-every`
steps, to follow training. final.pt holds the final weights and the training summary (steps, training and test KL,
whether the target was reached); it is the only file of a run kept in the repository.

Usage: python train.py --eps 0.05 --seed 0 --out runs/L4_d64/eps0.05_s0
"""
import argparse
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

import mrrxor
from model import Config, untrained

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--init-std", type=float, default=0.1,
                    help="std of the normal initialization of all weights")
parser.add_argument("--layers", type=int, default=4)
parser.add_argument("--d-model", type=int, default=64, help="residual-stream width; the MLPs have width 4 d_model")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eps", type=float, default=0.05, help="MRRXOR mixing probability")
parser.add_argument("--context", type=int, default=32)
parser.add_argument("--batch", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--warmup", type=int, default=1000)
parser.add_argument("--max-steps", type=int, default=100_000, help="length of the cosine schedule and step budget")
parser.add_argument("--kl-target", type=float, default=1e-4)
parser.add_argument("--window", type=int, default=500)
parser.add_argument("--log-every", type=int, default=1000)
parser.add_argument("--ckpt-every", type=int, default=5000)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
data_rng = torch.Generator().manual_seed(args.seed)
T = mrrxor.transition_tensor(args.eps)

cfg = Config(context=args.context, n_layers=args.layers, d_model=args.d_model, d_mlp=4 * args.d_model)
model = untrained(cfg, args.init_std, args.seed).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
    torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=args.warmup),
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.max_steps - args.warmup),
                                               eta_min=args.lr / 100),
], milestones=[args.warmup])

args.out.mkdir(parents=True, exist_ok=True)
test_tokens, test_optimal = mrrxor.test_set(T, args.context)


def test_kl() -> float:
    with torch.no_grad():
        return mrrxor.kl_to_model(test_optimal, model(test_tokens.to(device)))


# Saved with every checkpoint: the settings that define the run (the architecture is in "config").
RUN_SETTINGS = ("eps", "seed", "init_std", "context", "batch", "lr", "warmup", "max_steps", "kl_target", "window")


def save(name: str, step: int, **summary) -> None:
    torch.save({"config": vars(cfg), "args": {k: str(getattr(args, k)) for k in RUN_SETTINGS}, "step": step,
                "model": {key: value.cpu() for key, value in model.state_dict().items()}, **summary}, args.out / name)


recent_kl, started, step = deque(maxlen=args.window), time.time(), 0
while step < args.max_steps:
    tokens = mrrxor.sample(T, args.batch, args.context + 1, data_rng)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    optimal = mrrxor.optimal_predictions(T, inputs)  # (batch, context, 2), aligned with the logits

    logits = model(inputs.to(device))
    loss = F.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
    step += 1

    recent_kl.append(mrrxor.kl_to_model(optimal, logits))
    kl = sum(recent_kl) / len(recent_kl)
    done = len(recent_kl) == args.window and kl < args.kl_target
    if step % args.log_every == 0 or done:
        print(f"step {step:7d}  lr {scheduler.get_last_lr()[0]:.2e}  train KL {kl:.2e}  test KL {test_kl():.2e}"
              f"  ({step / (time.time() - started):.0f} steps/s)", flush=True)
    if step % args.ckpt_every == 0:
        save(f"step_{step:06d}.pt", step)
    if done:
        break

reached = len(recent_kl) == args.window and kl < args.kl_target
save("final.pt", step, train_kl=kl, test_kl=test_kl(), reached_target=reached, seconds=time.time() - started)
print(f"{'reached' if reached else 'did not reach'} train KL {args.kl_target:g} after {step} steps "
      f"(train KL {kl:.2e})", flush=True)
