"""Collect the probe results runs/<model>/eps<eps>/s<seed>.json into the markdown tables of results.md (also
printed). There is one section per split of the probe's rows (random; odd/even positions; first half to second half;
second half to first half), each with a table for uniform weighting and a collapsed one for process weighting. Rows
are models and mixing probabilities; entries are the mean ± standard deviation over seeds of the final test KL and of
the held-out R^2 of the belief-state probe (bold) and of its three controls (the last being the belief-state probe on
the untrained network). Below the tables: the number of seeds, if not 5."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

NAMES = {(4, 64, False): "large", (4, 64, True): "large, RoPE", (2, 32, False): "small",
         (2, 32, True): "small, RoPE"}  # labels used in the README
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


def fmt(values: list[float], show=number) -> str:
    return f"{show(mean(values))} ± {show(stdev(values))}" if len(values) > 1 else show(values[0])


def kl(value: float) -> str:
    return f"{value:.1e}".replace("e-0", "e-")


def rows():
    """(model, eps, results of its seeds) in table order."""
    for model in models:
        for eps in sorted((key[3] for key in runs if key[:3] == model), reverse=True):
            yield model, eps, runs[(*model, eps)]


def table(split: str, weighting: str) -> str:
    lines = ["| model | ε | test KL | " + " | ".join(column for column, _, _ in COLUMNS) + " |",
             "|---|" + "---:|" * (2 + len(COLUMNS))]
    for model, eps, rs in rows():
        cells = [fmt([r[f"{network}{split}{target}r2_{weighting}"] for r in rs]) for _, network, target in COLUMNS]
        lines.append(f"| {name(*model)} | {eps:g} | {fmt([r['test_kl'] for r in rs], kl)} | **{cells[0]}** | "
                     + " | ".join(cells[1:]) + " |")
    return "\n".join(lines)


counts = {len(rs) for _, _, rs in rows()}
if len(counts) == 1:
    notes = [f"Seeds per row: {counts.pop()}."] if counts != {5} else []
else:
    notes = [f"{name(*model)}, ε = {eps:g}: {len(rs)} seeds." for model, eps, rs in rows() if len(rs) != 5]
sections = [f"### {title}\n\n{table(split, 'uniform')}\n\n"
            f"<details>\n<summary>Process weighting</summary>\n\n{table(split, 'process')}\n\n</details>"
            for title, split in SPLITS]
text = "\n\n".join(sections + notes)
Path("results.md").write_text(text + "\n")
print(text)
