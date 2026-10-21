"""Collect runs/<model>/<run>/probe.json into markdown tables (printed, and written to results.md). For each model
size there is one table per weighting of histories (uniform, then process in a collapsed block), with one row per
mixing probability: the mean ± standard deviation over seeds of the held-out R^2 of the belief-state probe (bold)
and of its three controls. Rows with other than 5 seeds, and runs that did not reach the training-KL target, are
noted below the tables."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

columns = [("belief state", "r2"), ("next-token distribution", "next_r2"), ("shuffled belief state", "shuffled_r2"),
           ("untrained network", "init_r2")]

models = defaultdict(lambda: defaultdict(list))  # (layers, d_model) -> eps -> results
for path in sorted(Path("runs").glob("*/*/probe.json")):
    result = json.loads(path.read_text())
    models[result["layers"], result["d_model"]][result["eps"]].append(result)


def fmt(values: list[float]) -> str:
    return f"{mean(values):.2f} ± {stdev(values):.2f}" if len(values) > 1 else f"{values[0]:.2f}"


def table(rows: dict, weighting: str) -> str:
    lines = ["| ε | " + " | ".join(name for name, _ in columns) + " |", "|" + "---:|" * (1 + len(columns))]
    for eps in sorted(rows, reverse=True):
        cells = [fmt([r[f"{key}_{weighting}"] for r in rows[eps]]) for _, key in columns]
        lines.append(f"| {eps:g} | **{cells[0]}** | " + " | ".join(cells[1:]) + " |")
    return "\n".join(lines)


sections = []
for layers, d_model in sorted(models, reverse=True):
    rows = models[layers, d_model]
    notes = [f"ε = {eps:g}: {len(rows[eps])} seeds." for eps in sorted(rows, reverse=True) if len(rows[eps]) != 5]
    missed = [f"ε = {eps:g} ({n} of {len(rows[eps])} seeds)" for eps in sorted(rows, reverse=True)
              if (n := sum(not r["reached_target"] for r in rows[eps]))]
    if missed:
        notes.append(f"Did not reach the training-KL target: {'; '.join(missed)}.")
    sections.append(f"### {layers} layers, d_model {d_model}\n\n{table(rows, 'uniform')}\n\n"
                    f"<details>\n<summary>Process weighting</summary>\n\n{table(rows, 'process')}\n\n</details>"
                    + "".join(f"\n\n{note}" for note in notes))
text = "\n\n".join(sections)
Path("results.md").write_text(text + "\n")
print(text)
