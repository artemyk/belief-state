"""Collect the probe results runs/<model>/eps<eps>/s<seed>.json into the markdown tables of results.md (also
printed). There is one section per split of the probe's rows (random; odd/even positions; first half to second half;
second half to first half), each with a table for uniform weighting and a collapsed one for process weighting. Rows
are models and mixing probabilities; entries are the mean ± standard deviation over seeds of the held-out R^2 of the
belief-state probe (bold) and of its three controls (the last being the belief-state probe on the untrained network).
Runs that did not reach the training-KL target, and rows with other than 5 seeds, are noted below the tables."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

NAMES = {(4, 64, False): "large", (4, 64, True): "large, RoPE", (2, 32, False): "small"}  # labels used in the README
SPLITS = [("Random split", ""), ("Odd/even split", "parity_"), ("First half to second half", "early_"),
          ("Second half to first half", "late_")]
# (column, network prefix, target prefix): result keys are <network><split><target>r2_<weighting>
COLUMNS = [("belief state", "", ""), ("next-token distribution", "", "next_"),
           ("shuffled belief state", "", "shuffled_"), ("untrained network", "init_", "")]

runs = defaultdict(list)  # (layers, d_model, rope, eps) -> results
for path in sorted(Path("runs").glob("*/*/*.json")):
    result = json.loads(path.read_text())
    runs[result["layers"], result["d_model"], result["rope"], result["eps"]].append(result)
models = sorted({key[:3] for key in runs}, key=lambda m: (-m[0], -m[1], m[2]))  # larger first, RoPE after learned


def name(layers: int, d_model: int, rope: bool) -> str:
    return NAMES.get((layers, d_model, rope), f"{layers} layers, d_model {d_model}" + (", RoPE" if rope else ""))


def number(value: float) -> str:
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def fmt(values: list[float]) -> str:
    return f"{number(mean(values))} ± {number(stdev(values))}" if len(values) > 1 else number(values[0])


def table(split: str, weighting: str) -> str:
    lines = ["| model | ε | " + " | ".join(column for column, _, _ in COLUMNS) + " |",
             "|---|" + "---:|" * (1 + len(COLUMNS))]
    for model in models:
        for eps in sorted((key[3] for key in runs if key[:3] == model), reverse=True):
            rs = runs[(*model, eps)]
            cells = [fmt([r[f"{network}{split}{target}r2_{weighting}"] for r in rs]) for _, network, target in COLUMNS]
            lines.append(f"| {name(*model)} | {eps:g} | **{cells[0]}** | " + " | ".join(cells[1:]) + " |")
    return "\n".join(lines)


notes = [f"{name(*key[:3])}, ε = {key[3]:g}: {len(rs)} seeds." for key, rs in sorted(runs.items(), reverse=True)
         if len(rs) != 5]
missed = [f"{name(*key[:3])}, ε = {key[3]:g} ({n} of {len(rs)} seeds)"
          for key, rs in sorted(runs.items(), reverse=True) if (n := sum(not r["reached_target"] for r in rs))]
if missed:
    notes.append(f"Did not reach the training-KL target: {'; '.join(missed)}.")
sections = [f"### {title}\n\n{table(split, 'uniform')}\n\n"
            f"<details>\n<summary>Process weighting</summary>\n\n{table(split, 'process')}\n\n</details>"
            for title, split in SPLITS]
text = "\n\n".join(sections + notes)
Path("results.md").write_text(text + "\n")
print(text)
