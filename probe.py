"""Load checkpoints, run the belief-state probes, and save their results and provenance.

Usage: python probe.py runs/<model>/eps<eps>/s<seed>.pt [more runs] [--checkpoint final|latest|all|step_<step>]
  A run can also be given without the .pt extension.
  final: probes <run>.pt and, as a baseline, the untrained network (regenerated from the run's seed), and writes
         <run>.json
  step_<step>: probes <run>/step_<step>.pt and writes <run>/probe_step_<step>.json
  latest: the newest step checkpoint; all: every step checkpoint with missing or stale probe results
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

import probing
from model import Config, Transformer


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
        return steps
    return [checkpoint]


def result_metadata(path: Path, checkpoint: dict, settings: dict) -> dict:
    """Identify the exact checkpoint, recorded training recipe, and implementation used by the probe.

    The code hash covers the model, process and probe sources. It identifies the code used for probing;
    the checkpoint's training settings describe the original training run.
    """
    code = hashlib.sha256()
    for name in ("model.py", "mrrxor.py", "probing.py"):
        code.update(name.encode() + b"\0" + Path(__file__).with_name(name).read_bytes())
    # Stored checkpoints may serialize numeric arguments as strings. Keep numeric types in result metadata.
    training = {key: json.loads(value) if isinstance(value, str) else value
                for key, value in checkpoint["args"].items() if key != "context"}
    return {"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "config": checkpoint["config"], "training_settings": training,
            "probe_settings": {**settings, "relative_alphas": probing.ALPHAS.tolist(),
                               "cv": "leave-one-position-out, mean weighted MSE", "code_sha256": code.hexdigest()}}


def result_is_current(path: Path, metadata: dict) -> bool:
    """Reuse a cached probe only when its checkpoint, implementation, and settings match."""
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return isinstance(result, dict) and all(result.get(key) == value for key, value in metadata.items())


def format_r2(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


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
        metadata = result_metadata(checkpoint_path, checkpoint, settings)
        if args.checkpoint == "all" and result_is_current(results_path, metadata):
            continue
        result = {"step": checkpoint["step"], **metadata,
                  **probing.evaluate(model, metadata["training_settings"], settings, include_initial=name == "final")}
        results_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"{run.parent.parent.name}/{run.parent.name}/{run.name} ({name}, step {result['step']}): "
              f"test KL {result['test_kl']:.2e}, pooled held-out R^2 {format_r2(result['r2_uniform'])} (uniform), "
              f"{format_r2(result['r2_process'])} (process); next-token "
              f"{format_r2(result['next_r2_uniform'])}, {format_r2(result['next_r2_process'])}; shuffled "
              f"{format_r2(result['shuffled_r2_uniform'])}, {format_r2(result['shuffled_r2_process'])}"
              + (f"; untrained {format_r2(result['init_r2_uniform'])}, {format_r2(result['init_r2_process'])}"
                 if name == "final" else ""), flush=True)


if __name__ == "__main__":
    main()
