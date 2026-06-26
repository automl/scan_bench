import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def valid_files(root: Path, filename: str):
    for file in root.rglob(f"fold*/{filename}"):
        if (
                "fit_with_intermediate" in file.parts
                and "pred_with_intermediate" in file.parts
        ):
            yield file


def collect_learning_curves(root: Path) -> pd.DataFrame:
    rows = []

    for file in valid_files(root, "val_loss.json"):
        parts = file.relative_to(root).parts
        model = "/".join(parts[:-5])

        if model == "autogluon":
            continue

        seed = next(p for p in parts if p.startswith("seed="))
        fold = next(p for p in parts if p.startswith("fold"))

        for n_samples, metrics in json.loads(file.read_text()).items():
            n_samples = int(n_samples)

            if n_samples < 50 or n_samples == 1093:
                continue

            rows.append({
                "model": model,
                "seed": seed,
                "fold": fold,
                "n_samples": n_samples,
                **metrics,
            })

    return pd.DataFrame(rows)


def plot_learning_curves(df: pd.DataFrame, metric="rmse"):
    summary = (
        df.groupby(["model", "n_samples"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    groups = list(summary.groupby("model"))
    colors = plt.get_cmap("Dark2").colors

    plt.figure(figsize=(10, 6))

    for i, (model, group) in enumerate(groups):
        group = group.sort_values("n_samples")
        color = colors[i % len(colors)]

        plt.plot(
            group["n_samples"],
            group["mean"],
            label=model,
            color=color,
            linewidth=2,
        )

        plt.fill_between(
            group["n_samples"],
            group["mean"] - group["std"].fillna(0),
            group["mean"] + group["std"].fillna(0),
            color=color,
            alpha=0.15,
        )

    plt.xlabel("Number of samples")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} learning curves")
    plt.xscale("log")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root = Path("../../vlm/performance_surrogate/results")

    df = collect_learning_curves(root)
    metrics = [
        "rmse",
        "mae",
        "mdae",
        "marpd",
        "r2",
        "r",
        "spearman",
    ]

    for metric in metrics:
        plot_learning_curves(df, metric)
