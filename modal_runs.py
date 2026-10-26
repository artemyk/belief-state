"""Train and probe runs on Modal (modal.com) GPUs instead of locally.

Each container trains a group of runs on one GPU, then probes them on the CPU. By default a group is a single run:
with torch.compile's default mode, a large run on an L4 takes about 180 steps/s alone, and four concurrent runs about
40 steps/s each. Runs are written to the Modal volume belief-state-runs under
<tag>/runs/<model>/eps<eps>/s<seed>: the final checkpoint (.pt), the training log (.log) and the probe results (.json),
as run_sweep.sh writes them locally; no intermediate checkpoints. Copy them back with
  modal volume get --force belief-state-runs <tag>/runs .    # --force: overwrite existing files in runs/
and run summarize.py.

Usage:
  modal run modal_runs.py::benchmark --gpu T4     # a few thousand steps, 1 and 4 runs per GPU, to time a step
  modal run modal_runs.py::main                   # all four models, eps 0.05 and 0, seeds 0-4
  modal run modal_runs.py::main --seeds 0         # options: --seeds, --per-gpu, --gpu, --steps, --tag
The Starter plan runs at most 10 GPUs at a time; further groups wait for a free one.
"""
import subprocess
import time
from pathlib import Path

import modal

FILES = ("model.py", "mrrxor.py", "probe.py", "train.py")
MODELS = [(4, 64, False), (4, 64, True), (2, 32, False), (2, 32, True)]  # (layers, d_model, rope)

app = modal.App("belief-state")
image = modal.Image.debian_slim(python_version="3.12").apt_install("build-essential").pip_install(
    "torch==2.14.1", "numba==0.68.0", "numpy==2.5.4")  # build-essential: Triton compiles its launcher with gcc
for name in FILES:
    image = image.add_local_file(Path(__file__).parent / name, f"/root/belief-state/{name}")
volume = modal.Volume.from_name("belief-state-runs", create_if_missing=True)


def run_name(layers: int, d_model: int, rope: bool, eps: float, seed: int) -> str:
    return f"L{layers}_d{d_model}{'_rope' if rope else ''}/eps{eps:g}/s{seed}"


@app.function(image=image, gpu="L4", cpu=4, volumes={"/vol": volume}, timeout=4 * 3600)
def train_group(jobs: list[tuple], steps: int, tag: str, tail: int = 1) -> list[str]:
    """Train the runs jobs = [(layers, d_model, rope, eps, seed), ...] concurrently, then probe them. Returns the last
    tail log lines of each run and the wall time."""
    started = time.time()
    root = Path(f"/vol/{tag}/runs")
    processes = []
    for layers, d_model, rope, eps, seed in jobs:
        out = root / run_name(layers, d_model, rope, eps, seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        command = ["python", "train.py", "--layers", str(layers), "--d-model", str(d_model), "--eps", str(eps),
                   "--seed", str(seed), "--steps", str(steps), "--ckpt-every", str(10 * steps), "--out", str(out)]
        log = open(f"{out}.log", "w")
        processes.append((out, log, subprocess.Popen(command + ["--rope"] * rope, cwd="/root/belief-state",
                                                     stdout=log, stderr=subprocess.STDOUT)))
    for _, log, process in processes:
        process.wait()
        log.close()
    trained = time.time() - started
    finished = [str(out) + ".pt" for out, _, process in processes if process.returncode == 0]
    if finished:
        subprocess.run(["python", "probe.py", *finished], cwd="/root/belief-state", check=True)
    volume.commit()
    lines = [f"{out.relative_to(root)}: {line}" for out, _, _ in processes
             for line in Path(f"{out}.log").read_text().strip().splitlines()[-tail:]]
    return lines + [f"{len(jobs)} runs: trained in {trained:.0f} s, probed in {time.time() - started - trained:.0f} s"]


@app.local_entrypoint()
def main(seeds: str = "0,1,2,3,4", per_gpu: int = 1, gpu: str = "L4", steps: int = 100_000,
         tag: str = "masked_bos"):
    """Every model, eps 0.05 and 0, and the given seeds, in groups of per_gpu runs; each group mixes large and small
    runs, so that the groups finish at about the same time."""
    jobs = [(*model, eps, int(seed)) for model in MODELS for eps in (0.05, 0.0) for seed in seeds.split(",")]
    count = -(-len(jobs) // per_gpu)  # number of groups
    groups = [jobs[i::count] for i in range(count)]
    for lines in train_group.with_options(gpu=gpu).starmap([(group, steps, tag) for group in groups]):
        print("\n".join(lines), flush=True)


@app.local_entrypoint()
def benchmark(steps: int = 3000, gpu: str = "L4"):
    """Large RoPE runs, eps 0.05, for a few thousand steps on one kind of GPU: one run alone, and four concurrent
    runs."""
    calls = {count: train_group.with_options(gpu=gpu).spawn([(4, 64, True, 0.05, seed) for seed in range(count)],
                                                            steps, f"benchmark_{gpu}_{count}", tail=4)
             for count in (1, 4)}
    for count, call in calls.items():
        print(f"--- {gpu}, {count} concurrent run(s)\n" + "\n".join(call.get()), flush=True)
