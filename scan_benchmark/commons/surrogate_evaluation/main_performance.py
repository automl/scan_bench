import json
from pathlib import Path

import pandas as pd


def valid_files(root: Path, filename: str):
    for file in root.rglob(f"fold*/{filename}"):
        if (
                "fit_with_intermediate" in file.parts
                and "pred_with_intermediate" in file.parts
        ):
            yield file


def get_model_name(file: Path, root: Path) -> str:
    return "/".join(file.relative_to(root).parts[:-5])


def summarize_metrics(root: Path) -> pd.DataFrame:
    rows = []

    for file in valid_files(root, "val_val_loss.json"):
        data = json.loads(file.read_text())
        last_metrics = data[max(data, key=lambda key: int(key))]

        rows.append({
            "model": get_model_name(file, root),
            **last_metrics,
        })

    return pd.DataFrame(rows).groupby("model").agg(["mean", "std"])


def summarize_metrics_by_bin(root: Path) -> pd.DataFrame:
    rows = []

    for file in valid_files(root, "val_val_loss_by_bin.json"):
        data = json.loads(file.read_text())["metrics_by_bin"]

        for bin_name, metrics in data.items():
            rows.append({
                "model": get_model_name(file, root),
                "bin": bin_name,
                **metrics,
            })

    return pd.DataFrame(rows).groupby(["model", "bin"]).agg(["mean", "std"])


def summarize_metrics_by_bin_top_performing(root: Path) -> pd.DataFrame:
    rows = []

    for file in valid_files(root, "val_val_loss_top_performing_per_bin.json"):
        data = json.loads(file.read_text())["metrics_by_bin"]

        for bin_name, metrics in data.items():
            rows.append({
                "model": get_model_name(file, root),
                "bin": bin_name,
                **metrics,
            })

    return pd.DataFrame(rows).groupby(["model", "bin"]).agg(["mean", "std"])


if __name__ == "__main__":
    root = Path("../../tabpfn/performance_surrogate/results")

    summarize_metrics(root).to_csv("results_summary.csv")
    summarize_metrics_by_bin(root).to_csv("results_summary_by_bin.csv")
    summarize_metrics_by_bin_top_performing(root).to_csv("results_summary_by_bin_top_performing.csv")
