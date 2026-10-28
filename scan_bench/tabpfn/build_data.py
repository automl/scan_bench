import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader


DEFAULT_RUNS_ROOT = "/work/dlc2workfs1/alipourn-tabpfn/runs/scan"
DEFAULT_SOURCES = ["random_search:v1", "random_search_v2:v2", "random_search_v3:v3"]

REAL_DATASETS = ["iris", "wdbc", "wine"]
VAL_SETS = ["small", "medium"]

OUTPUT_COLUMNS = [
    "config_id",
    "real_data/nll",
    "real_data/roc_auc",
    "val/large/val_loss",
    "val/medium/val_loss",
    "val/small/val_loss",
    "val/val_loss",
    "total_flops",
    "total_cells",
    "config/effective_batch_size",
    "config/lr",
    "config/max_features",
    "config/model_config.embedding_size",
    "config/model_config.num_layers",
    "config/num_datapoints_max",
    "parameters",
    "step",
    "is_final",
]

REQUIRED_COLUMNS = [
    "val/val_loss",
    "total_flops",
    "total_cells",
    "config/effective_batch_size",
    "config/lr",
    "config/max_features",
    "config/model_config.embedding_size",
    "config/model_config.num_layers",
    "config/num_datapoints_max",
]

DEFAULT_DIVERGENCE_THRESHOLD = 2.6

INTERMEDIATE_REPORT_PATTERN = re.compile(r"report_(\d+)\.yaml")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=SafeLoader)


def mean_or_nan(values: list) -> float:
    if any(v is None or not math.isfinite(float(v)) for v in values):
        return np.nan
    return float(np.mean(values))


def config_columns(config: dict) -> dict:
    return {
        "config/effective_batch_size": config["effective_batch_size"],
        "config/lr": config["lr"],
        "config/max_features": config["max_features"],
        "config/model_config.embedding_size": config["model_config.embedding_size"],
        "config/model_config.num_layers": config["model_config.num_layers"],
        "config/num_datapoints_max": config["num_datapoints_max"],
    }


def metric_columns(extra: dict) -> dict:
    real_data = [extra.get(f"real_data/{name}", {}) for name in REAL_DATASETS]
    large = extra.get("val/large", {}).get("val_loss", np.nan)
    small = extra.get("val/small", {}).get("val_loss")
    medium = extra.get("val/medium", {}).get("val_loss")

    return {
        "real_data/nll": mean_or_nan([d.get("nll") for d in real_data]),
        "real_data/roc_auc": mean_or_nan([d.get("roc_auc") for d in real_data]),
        "val/large/val_loss": large,
        "val/medium/val_loss": medium if medium is not None else np.nan,
        "val/small/val_loss": small if small is not None else np.nan,
        # val/large is only evaluated at the end of training, so the target is
        # defined on small and medium for both final and intermediate rows.
        "val/val_loss": mean_or_nan([small, medium]),
        "parameters": extra.get("parameters", np.nan),
    }


def read_final_step(config_dir: Path, intermediate_steps: list[int]) -> int:
    run_info_path = config_dir / "run_info.yaml"
    if run_info_path.exists():
        step = load_yaml(run_info_path).get("step")
        if step is not None:
            return int(step)
    return max(intermediate_steps, default=0) + 1


def read_config_rows(config_dir: Path, prefix: str) -> list[dict]:
    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        return []

    config = load_yaml(config_path)
    config_id = f"{prefix}_{config_dir.name.rsplit('_', 1)[1]}"
    base = {"config_id": config_id, **config_columns(config)}

    intermediate_paths = {
        int(m.group(1)): config_dir / f.name
        for f in config_dir.iterdir()
        if (m := INTERMEDIATE_REPORT_PATTERN.fullmatch(f.name))
    }

    rows = []
    final_step = None

    final_path = config_dir / "report.yaml"
    if final_path.exists():
        report = load_yaml(final_path)
        if report.get("reported_as") == "success" and report.get("err") is None:
            final_step = read_final_step(config_dir, list(intermediate_paths))
            extra = report["extra"]
            rows.append({
                **base,
                **metric_columns(extra),
                "total_flops": extra["flops"],
                "total_cells": config["num_cells_training"],
                "step": final_step,
                "is_final": True,
            })

    for step, path in sorted(intermediate_paths.items()):
        if step == final_step:
            continue
        report = load_yaml(path)
        extra = report["extra"]
        rows.append({
            **base,
            **metric_columns(extra),
            "total_flops": extra["total_flops"],
            "total_cells": extra["total_cells"],
            "step": int(report.get("step", step)),
            "is_final": False,
        })

    return rows


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    values = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(values.to_numpy(dtype=float))

    for col, n_bad in zip(REQUIRED_COLUMNS, (~finite_mask).sum(axis=0)):
        if n_bad:
            print(f"{n_bad} rows with missing or non-finite {col}.")

    keep = finite_mask.all(axis=1)
    filtered_df = df[keep].reset_index(drop=True)

    removed_configs = sorted(set(df["config_id"]) - set(filtered_df["config_id"]))
    if len(filtered_df) < len(df):
        print(f"\nRemoved {len(df) - len(filtered_df)} incomplete rows.")
    if removed_configs:
        print(f"WARNING: removed all rows for {len(removed_configs)} configs: {removed_configs}")

    return filtered_df


def drop_diverged(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # High-lr runs often spike early and recover, so only runs whose last
    # checkpoint is still above the threshold count as diverged.
    last_loss = df.sort_values("step").groupby("config_id")["val/val_loss"].last()
    diverged = last_loss[last_loss > threshold].index
    diverged_mask = df["config_id"].isin(diverged)
    spike_mask = ~diverged_mask & (df["val/val_loss"] > threshold)

    print(
        f"\nRemoved {len(diverged)} diverged configs ({int(diverged_mask.sum())} rows) "
        f"with last val/val_loss > {threshold}."
    )
    print(
        f"Removed {int(spike_mask.sum())} spike rows with val/val_loss > {threshold} "
        f"from {df.loc[spike_mask, 'config_id'].nunique()} recovered configs."
    )

    dropped_df = df[diverged_mask | spike_mask].copy()
    dropped_df["drop_reason"] = np.where(diverged_mask[dropped_df.index], "diverged", "loss_spike")

    return df[~(diverged_mask | spike_mask)], dropped_df


def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        _source=df["config_id"].str.split("_").str[0],
        _config_num=df["config_id"].str.split("_").str[1].astype(int),
    )
    return (
        df.sort_values(["_source", "_config_num", "step"])
        .drop(columns=["_source", "_config_num"])
        .reset_index(drop=True)
    )


def print_summary(df: pd.DataFrame) -> None:
    df = df.assign(source=df["config_id"].str.split("_").str[0])
    finished = df.groupby("config_id")["is_final"].any()
    summary = pd.DataFrame({
        "configs": df.groupby("source")["config_id"].nunique(),
        "finished_configs": finished.groupby(finished.index.str.split("_").str[0]).sum(),
        "final_rows": df[df["is_final"]].groupby("source").size(),
        "intermediate_rows": df[~df["is_final"]].groupby("source").size(),
    }).fillna(0).astype(int)
    summary.loc["total"] = summary.sum()
    print("\nSummary:")
    print(summary)


def build_dataframe(
        runs_root: Path,
        sources: list[str],
        divergence_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for source in sources:
        dirname, prefix = source.split(":")
        config_dirs = sorted((runs_root / dirname / "configs").glob("config_*"))
        print(f"Reading {len(config_dirs)} config dirs from {dirname} as {prefix}")
        for config_dir in config_dirs:
            rows.extend(read_config_rows(config_dir, prefix))

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df = drop_incomplete_rows(df)
    df, dropped_df = drop_diverged(df, divergence_threshold)
    df = sort_rows(df)
    dropped_df = sort_rows(dropped_df)

    duplicated = df.duplicated(subset=["config_id", "step"])
    if duplicated.any():
        raise ValueError(f"Found {int(duplicated.sum())} duplicated (config_id, step) rows.")

    return df, dropped_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default=DEFAULT_RUNS_ROOT)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        help="Run directories with their config id prefix, as <dirname>:<prefix>.",
    )
    parser.add_argument(
        "--divergence_threshold",
        type=float,
        default=DEFAULT_DIVERGENCE_THRESHOLD,
        help="Configs whose last val/val_loss exceeds this are dropped, as are single rows above it.",
    )
    parser.add_argument("--output_csv", default="scan_bench/tabpfn/data.csv")
    parser.add_argument(
        "--diverged_csv",
        default="scan_bench/tabpfn/data_diverged.csv",
        help="Rows removed by the divergence filter, with a drop_reason column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df, dropped_df = build_dataframe(Path(args.runs_root), args.sources, args.divergence_threshold)
    print_summary(df)

    for path, out_df in [(Path(args.output_csv), df), (Path(args.diverged_csv), dropped_df)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(path, index=False)
        print(f"\nWrote {len(out_df)} rows to {path}")


if __name__ == "__main__":
    main()
