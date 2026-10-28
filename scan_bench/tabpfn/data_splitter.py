import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_log_bins(
        df: pd.DataFrame,
        flops_col: str,
        num_bins: int = 4,
):
    log_x = np.log10(df[flops_col].to_numpy(dtype=float))

    bins = np.unique(np.quantile(log_x, np.linspace(0, 1, num_bins + 1)))
    labels = [f"bin_{i}" for i in range(len(bins) - 1)]

    df = df.copy()
    df["flops_bin"] = pd.cut(
        log_x,
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(str)

    return df, bins, labels


def format_bin_boundaries(bins) -> list[str]:
    lines = []
    for i in range(len(bins) - 1):
        low_log = bins[i]
        high_log = bins[i + 1]
        lines.append(
            f"bin_{i}: "
            f"log10 [{low_log:.2f}, {high_log:.2f}]  |  "
            f"FLOPs [{10 ** low_log:.2e}, {10 ** high_log:.2e}]"
        )
    return lines


def save_bin_boundaries(bins, filepath: Path) -> None:
    with open(filepath, "w") as f:
        f.write("Bin boundaries:\n\n")
        for line in format_bin_boundaries(bins):
            f.write(line + "\n")


def print_bin_counts(df: pd.DataFrame, col: str = "flops_bin") -> None:
    counts = df[col].value_counts().sort_index()
    print("\nCounts per bin:")
    print(counts)


def plot_gflops_distribution(df, flops_col, bins, filepath: Path) -> None:
    x = df[flops_col]

    plt.figure(figsize=(8, 2))
    plt.scatter(x, [0] * len(x), alpha=0.6)

    for b in 10 ** bins:
        plt.axvline(b, linestyle="--")

    plt.xscale("log")
    plt.yticks([])
    plt.xlabel("FLOPs")
    plt.title("Log10-spaced bins")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def get_last_eval_per_config(df: pd.DataFrame) -> pd.DataFrame:
    # The final report for finished runs, the latest checkpoint for unfinished ones.
    return (
        df.sort_values(["config_id", "step"])
        .drop_duplicates(subset="config_id", keep="last")
        .reset_index(drop=True)
    )


def enrich_with_all_eval_steps(
        full_df: pd.DataFrame,
        selected_configs_df: pd.DataFrame,
) -> pd.DataFrame:
    enriched_df = full_df.merge(
        selected_configs_df[["config_id", "flops_bin"]],
        on="config_id",
        how="inner",
    )
    return enriched_df.sort_values(["config_id", "step"]).reset_index(drop=True)


def prepare_splits(
        full_df: pd.DataFrame,
        flops_col: str,
        output_dir: Path,
        num_bins: int = 4,
        num_folds: int = 5,
        random_state: int = 42,
        make_plot: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    last_eval_df = get_last_eval_per_config(full_df)
    last_eval_df, bins, labels = create_log_bins(
        df=last_eval_df,
        flops_col=flops_col,
        num_bins=num_bins,
    )

    print("\nBin boundaries:")
    print("\n".join(format_bin_boundaries(bins)))
    save_bin_boundaries(bins, output_dir / "bin_boundaries.txt")
    print_bin_counts(last_eval_df)

    if make_plot:
        plot_gflops_distribution(
            df=last_eval_df,
            flops_col=flops_col,
            bins=bins,
            filepath=output_dir / "flops_bins.png",
        )

    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_state,
    )

    folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(last_eval_df, last_eval_df["flops_bin"]),
            start=1,
    ):
        train_last_df = last_eval_df.iloc[train_idx]
        test_last_df = last_eval_df.iloc[test_idx]

        print(f"\n--- Fold {fold_idx} Train split ---")
        print_bin_counts(train_last_df)

        print(f"\n--- Fold {fold_idx} Test split ---")
        print_bin_counts(test_last_df)

        train_df = enrich_with_all_eval_steps(full_df, train_last_df)
        test_df = enrich_with_all_eval_steps(full_df, test_last_df)

        train_df.to_csv(output_dir / f"train_fold_{fold_idx}.csv", index=False)
        test_df.to_csv(output_dir / f"test_fold_{fold_idx}.csv", index=False)

        if fold_idx == 1:
            train_df.to_csv(output_dir / "train.csv", index=False)
            test_df.to_csv(output_dir / "test.csv", index=False)

        folds.append((train_df, test_df))

    return folds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="scan_bench/tabpfn/data.csv")
    parser.add_argument("--output_dir", default="scan_bench/tabpfn/performance_surrogate/splits")
    parser.add_argument("--flops_col", default="total_flops")
    parser.add_argument("--num_bins", type=int, default=4)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepare_splits(
        full_df=pd.read_csv(args.input_csv),
        flops_col=args.flops_col,
        output_dir=Path(args.output_dir),
        num_bins=args.num_bins,
        num_folds=args.num_folds,
        random_state=args.seed,
        make_plot=args.plot,
    )


if __name__ == "__main__":
    main()
