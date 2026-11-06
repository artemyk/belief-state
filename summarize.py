"""Validate the probe results runs/<model>/eps<eps>/s<seed>.json and print Markdown tables to standard output.
Checkpoint hashes must match the accompanying .pt files. All runs must use the same probe settings; within each
table row, the model and training settings must also agree, apart from the training seed.
There is one section per mixing probability eps, with a row per model (size and kind of position embedding) in each
table, and a bold title above each table. The first table has the final test KL and the held-out R^2 of the
belief-state probe for each split of the probe's rows (random; odd/even positions; first half to second half; second
half to first half); the second has its three controls for the random split (next-token distribution, shuffled belief
state, and the belief-state probe on the untrained network). Both weight histories uniformly. Collapsed below them
are the same two tables with process weighting and the controls for the held-out splits. Entries are means over seeds, with the standard deviation for the
belief state (joined by no-break spaces, so that it does not wrap). Below the tables: the number of seeds, if not 5."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

SIZES = {(4, 64): "large", (2, 32): "small"}  # labels used in the README
SPLITS = [("random", ""), ("odd/even", "parity_"), ("1st → 2nd half", "early_"), ("2nd → 1st half", "late_")]
# (column, network prefix, target prefix): result keys are <network><split><target>r2_<weighting>
CONTROLS = [("next-token distribution", "", "next_"), ("shuffled belief state", "", "shuffled_"),
            ("untrained network", "init_", "")]

def load_runs(paths: list[Path]) -> dict:
    """Verify checkpoint identities and reject incompatible settings or duplicate seeds before averaging."""
    runs = defaultdict(list)
    common_probe = None
    row_settings = {}
    for path in paths:
        result = json.loads(path.read_text())
        if not {"step", "checkpoint_sha256", "config", "training_settings", "probe_settings"} <= result.keys():
            raise ValueError(f"{path}: missing provenance; rerun probe.py for this checkpoint.")
        checkpoint = path.with_suffix(".pt")
        if not checkpoint.exists() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != result["checkpoint_sha256"]:
            raise ValueError(f"{path}: checkpoint missing or hash mismatch; rerun probe.py for the current checkpoint.")
        if common_probe is not None and result["probe_settings"] != common_probe:
            raise ValueError(f"{path}: incompatible probe settings.")
        common_probe = result["probe_settings"]
        config, training = result["config"], result["training_settings"]
        key = config["n_layers"], config["d_model"], config["rope"], training["eps"]
        settings = (config, {k: v for k, v in training.items() if k != "seed"}, result["step"])
        if key in row_settings and settings != row_settings[key]:
            raise ValueError(f"{path}: incompatible model or training settings within the same table row.")
        if any(previous["training_settings"]["seed"] == training["seed"] for previous in runs[key]):
            raise ValueError(f"{path}: duplicate training seed {training['seed']} within the same table row.")
        row_settings[key] = settings
        runs[key].append(result)
    return runs


def name(layers: int, d_model: int, rope: bool) -> str:
    return SIZES.get((layers, d_model), f"{layers} layers, d_model {d_model}") + (", RoPE" if rope else "")


def number(value: float | None) -> str:
    if value is None:
        return "undefined"
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def mean_score(values: list[float | None]) -> float | None:
    """An aggregate score is undefined if any seed's score is undefined."""
    return None if any(value is None for value in values) else mean(values)


def fmt(values: list[float | None]) -> str:
    if any(value is None for value in values):
        return "undefined"
    return f"{number(mean(values))} ± {number(stdev(values))}" if len(values) > 1 else number(values[0])


def kl(value: float) -> str:
    return f"{value:.1e}".replace("e-0", "e-")


def rows(runs: dict, eps: float):
    """(model, results of its seeds) in table order."""
    models = sorted({key[:3] for key in runs}, key=lambda m: (m[2], -m[0], -m[1]))
    for model in models:
        if (*model, eps) in runs:
            yield model, runs[(*model, eps)]


def belief_table(runs: dict, eps: float, weighting: str) -> str:
    lines = ["| model | test KL | " + " | ".join(title for title, _ in SPLITS) + " |",
             "|---|" + "---:|" * (1 + len(SPLITS))]
    for model, rs in rows(runs, eps):
        cells = [fmt([r[f"{split}r2_{weighting}"] for r in rs]) for _, split in SPLITS]
        lines.append(f"| {name(*model)} | {kl(mean(r['test_kl'] for r in rs))} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def controls_table(runs: dict, eps: float, weighting: str, splits: list) -> str:
    """The controls for the given splits, with a column naming the split if there are several."""
    split_column = len(splits) > 1
    lines = ["| " + "split | " * split_column + "model | " + " | ".join(c for c, _, _ in CONTROLS) + " |",
             "|" + "---|" * (1 + split_column) + "---:|" * len(CONTROLS)]
    for title, split in splits:
        for model, rs in rows(runs, eps):
            cells = [number(mean_score([r[f"{network}{split}{target}r2_{weighting}"] for r in rs]))
                     for _, network, target in CONTROLS]
            lines.append("| " + f"{title} | " * split_column + f"{name(*model)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def titled(runs: dict, weighting: str, eps: float) -> list[str]:
    """The belief-state table and the controls for the random split, each under a bold title."""
    return [f"**Test KL and belief-state R² by probe split**\n\n{belief_table(runs, eps, weighting)}",
            f"**R² of the controls, random split**\n\n{controls_table(runs, eps, weighting, SPLITS[:1])}"]


def section(runs: dict, eps: float) -> str:
    parts = [f"### ε = {eps:g}", *titled(runs, "uniform", eps)]
    parts.append("<details>\n<summary>Process weighting</summary>\n\n" + "\n\n".join(titled(runs, "process", eps))
                 + "\n\n</details>")
    parts.append(f"<details>\n<summary>R² of the controls, held-out splits</summary>\n\n"
                 f"{controls_table(runs, eps, 'uniform', SPLITS[1:])}\n\n</details>")
    return "\n\n".join(parts)


def render_summary(runs: dict) -> str:
    """Render validated results as Markdown without reading, writing, or printing anything."""
    eps_values = sorted({key[3] for key in runs}, reverse=True)
    counts = {len(rs) for eps in eps_values for _, rs in rows(runs, eps)}
    if len(counts) == 1:
        notes = [f"Seeds per row: {counts.pop()}."] if counts != {5} else []
    else:
        notes = [f"{name(*model)}, ε = {eps:g}: {len(rs)} seeds."
                 for eps in eps_values for model, rs in rows(runs, eps) if len(rs) != 5]
    return "\n\n".join([section(runs, eps) for eps in eps_values] + notes)


def main() -> None:
    runs = load_runs(sorted(Path("runs").glob("*/*/*.json")))
    print(render_summary(runs))


if __name__ == "__main__":
    main()
