"""Collect the probe results runs/<model>/eps<eps>/s<seed>.json into the markdown tables of results.md (also
printed). There is one section per mixing probability eps, with a row per model (size and kind of position
embedding) in each table. The first table has the final test KL and the held-out R^2 of the belief-state probe for
each split of the probe's rows (random; odd/even positions; first half to second half; second half to first half);
the second has its three controls for the random split (next-token distribution, shuffled belief state, and the
belief-state probe on the untrained network). Both weight histories uniformly. Collapsed below them are the same two
tables with process weighting (for eps > 0 only: for eps = 0 the two weightings give nearly the same results) and the
controls for the held-out splits. Entries are means over seeds, with the standard deviation for the belief state
(joined by no-break spaces, so that it does not wrap). Below the tables: the number of seeds, if not 5."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

SIZES = {(4, 64): "large", (2, 32): "small"}  # labels used in the README
SPLITS = [("random", ""), ("odd/even", "parity_"), ("1st → 2nd half", "early_"), ("2nd → 1st half", "late_")]
# (column, network prefix, target prefix): result keys are <network><split><target>r2_<weighting>
CONTROLS = [("next-token distribution", "", "next_"), ("shuffled belief state", "", "shuffled_"),
            ("untrained network", "init_", "")]

runs = defaultdict(list)  # (layers, d_model, rope, eps) -> results
for path in sorted(Path("runs").glob("*/*/*.json")):
    result = json.loads(path.read_text())
    runs[result["layers"], result["d_model"], result["rope"], result["eps"]].append(result)
models = sorted({key[:3] for key in runs}, key=lambda m: (m[2], -m[0], -m[1]))  # learned before RoPE, larger first
eps_values = sorted({key[3] for key in runs}, reverse=True)


def name(layers: int, d_model: int, rope: bool) -> str:
    return SIZES.get((layers, d_model), f"{layers} layers, d_model {d_model}") + (", RoPE" if rope else "")


def number(value: float) -> str:
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def fmt(values: list[float]) -> str:
    return f"{number(mean(values))} ± {number(stdev(values))}" if len(values) > 1 else number(values[0])


def kl(value: float) -> str:
    return f"{value:.1e}".replace("e-0", "e-")


def rows(eps: float):
    """(model, results of its seeds) in table order."""
    for model in models:
        if (*model, eps) in runs:
            yield model, runs[(*model, eps)]


def belief_table(eps: float, weighting: str) -> str:
    lines = ["| model | test KL | " + " | ".join(title for title, _ in SPLITS) + " |",
             "|---|" + "---:|" * (1 + len(SPLITS))]
    for model, rs in rows(eps):
        cells = [fmt([r[f"{split}r2_{weighting}"] for r in rs]) for _, split in SPLITS]
        lines.append(f"| {name(*model)} | {kl(mean(r['test_kl'] for r in rs))} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def controls_table(eps: float, weighting: str, splits: list) -> str:
    """The controls for the given splits, with a column naming the split if there are several."""
    split_column = len(splits) > 1
    lines = ["| " + "split | " * split_column + "model | " + " | ".join(c for c, _, _ in CONTROLS) + " |",
             "|" + "---|" * (1 + split_column) + "---:|" * len(CONTROLS)]
    for title, split in splits:
        for model, rs in rows(eps):
            cells = [number(mean(r[f"{network}{split}{target}r2_{weighting}"] for r in rs))
                     for _, network, target in CONTROLS]
            lines.append("| " + f"{title} | " * split_column + f"{name(*model)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def section(eps: float) -> str:
    parts = [f"### ε = {eps:g}", belief_table(eps, "uniform"), controls_table(eps, "uniform", SPLITS[:1])]
    if eps > 0:
        parts.append(f"<details>\n<summary>Process weighting</summary>\n\n{belief_table(eps, 'process')}\n\n"
                     f"{controls_table(eps, 'process', SPLITS[:1])}\n\n</details>")
    parts.append(f"<details>\n<summary>Controls for the held-out splits</summary>\n\n"
                 f"{controls_table(eps, 'uniform', SPLITS[1:])}\n\n</details>")
    return "\n\n".join(parts)


counts = {len(rs) for eps in eps_values for _, rs in rows(eps)}
if len(counts) == 1:
    notes = [f"Seeds per row: {counts.pop()}."] if counts != {5} else []
else:
    notes = [f"{name(*model)}, ε = {eps:g}: {len(rs)} seeds."
             for eps in eps_values for model, rs in rows(eps) if len(rs) != 5]
text = "\n\n".join([section(eps) for eps in eps_values] + notes)
Path("results.md").write_text(text + "\n")
print(text)
