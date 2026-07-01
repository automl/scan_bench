import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold

FEATURE_COLS = [
    "config_id",
    "global_batch_size",
    "beta1",
    "beta2",
    "eps",
    "lr",
    "text_width",
    "vision_width",
    "warmup_fraction",
    "wd",
    "total_samples_planned",
]


def save_bin_boundaries(bins, labels=None, filepath="bin_boundaries.txt"):
    with open(filepath, "w") as f:
        f.write("Bin boundaries:\n\n")

        for i in range(len(bins) - 1):
            low_log = bins[i]
            high_log = bins[i + 1]

            low = 10 ** low_log
            high = 10 ** high_log

            label = labels[i] if labels is not None else f"bin_{i}"

            f.write(
                f"{label}: "
                f"log10 [{low_log:.2f}, {high_log:.2f}]  |  "
                f"FLOPs [{low:.2e}, {high:.2e}]\n"
            )


def print_bin_counts(df: pd.DataFrame, col: str = "flops_bin") -> None:
    counts = df[col].value_counts().sort_index()
    print("\nCounts per bin:")
    print(counts)


def plot_gflops_distribution(df: pd.DataFrame, flops_col: str, bins):
    x = df[flops_col]

    plt.figure(figsize=(8, 2))
    plt.scatter(x, [0] * len(x), alpha=0.6)

    for b in 10 ** bins:
        plt.axvline(b, linestyle="--")

    plt.xscale("log")
    plt.yticks([])
    plt.xlabel("FLOPs")
    plt.title("Log10-spaced bins")
    plt.show()


def build_failure_labels(df: pd.DataFrame) -> pd.DataFrame:
    fail_df = (
        df.groupby("config_id", as_index=False)["epoch_diverged"]
        .any()
        .rename(columns={"epoch_diverged": "failed"})
    )
    fail_df["failed"] = fail_df["failed"].astype(int)
    return fail_df


def keep_only_nonfailed_last_epoch(df: pd.DataFrame) -> pd.DataFrame:
    fail_df = build_failure_labels(df)

    df = df.merge(fail_df, on="config_id", how="left")
    df_nonfailed = df[df["failed"] == 0].copy()

    df_last = (
        df_nonfailed
        .sort_values(["config_id", "epoch"])
        .drop_duplicates(subset="config_id", keep="last")
        .reset_index(drop=True)
    )

    return df_last


def enrich_with_all_epochs(
        full_df: pd.DataFrame,
        selected_configs_df: pd.DataFrame,
        keep_only_nonfailed: bool = True,
) -> pd.DataFrame:
    selected_config_ids = selected_configs_df["config_id"].unique()
    enriched_df = full_df[full_df["config_id"].isin(selected_config_ids)].copy()

    if "flops_bin" in selected_configs_df.columns:
        enriched_df = enriched_df.merge(
            selected_configs_df[["config_id", "flops_bin"]].drop_duplicates(),
            on="config_id",
            how="left",
        )

    if keep_only_nonfailed:
        fail_df = build_failure_labels(full_df)
        enriched_df = enriched_df.merge(fail_df, on="config_id", how="left")
        enriched_df = enriched_df[enriched_df["failed"] == 0].copy()

    enriched_df = (
        enriched_df
        .sort_values(["config_id", "epoch"])
        .reset_index(drop=True)
    )

    return enriched_df


def prepare_performance_predictor_data(
        full_df: pd.DataFrame,
        df: pd.DataFrame,
        flops_col: str,
        num_bins: int = 4,
        num_folds: int = 5,
        random_state: int = 42,
):
    x = df[flops_col].values
    log_x = np.log10(x)

    bins = np.quantile(log_x, np.linspace(0, 1, num_bins + 1))

    bins = np.unique(bins)

    labels = [f"bin_{i}" for i in range(len(bins) - 1)]

    save_bin_boundaries(bins, labels, filepath="performance_surrogate/splits/bin_boundaries.txt")

    df = df.copy()
    df["flops_bin"] = pd.cut(
        log_x,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    plot_gflops_distribution(df, flops_col=flops_col, bins=bins)

    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_state,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(df, df["flops_bin"]),
            start=1,
    ):
        train_last_df = df.iloc[train_idx].copy()
        test_last_df = df.iloc[test_idx].copy()

        print(f"\nFold {fold_idx} performance train split:")
        print_bin_counts(train_last_df)

        print(f"\nFold {fold_idx} performance test split:")
        print_bin_counts(test_last_df)

        train_df = enrich_with_all_epochs(full_df, train_last_df)
        test_df = enrich_with_all_epochs(full_df, test_last_df)

        train_output_csv = f"performance_surrogate/splits/train_fold_{fold_idx}.csv"
        test_output_csv = f"performance_surrogate/splits/test_fold_{fold_idx}.csv"

        train_df.to_csv(train_output_csv, index=False)
        test_df.to_csv(test_output_csv, index=False)


def build_feature_table(
        df: pd.DataFrame,
        feature_cols: list[str],
) -> pd.DataFrame:
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    features_df = (
        df[feature_cols]
        .sort_values("config_id")
        .drop_duplicates(subset="config_id", keep="first")
        .reset_index(drop=True)
    )

    return features_df


def build_full_failure_dataset(
        df: pd.DataFrame,
        feature_cols: list[str],
) -> pd.DataFrame:
    fail_df = build_failure_labels(df)
    features_df = build_feature_table(df, feature_cols)

    result = features_df.merge(fail_df, on="config_id", how="inner")
    result = result.sort_values("config_id").reset_index(drop=True)

    return result


def split_failed_configs(
        full_failure_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    failed_df = full_failure_df[full_failure_df["failed"] == 1].copy()

    train_failed_df, test_failed_df = train_test_split(
        failed_df,
        test_size=test_size,
        random_state=random_state,
    )

    train_failed_df = train_failed_df.sort_values("config_id").reset_index(drop=True)
    test_failed_df = test_failed_df.sort_values("config_id").reset_index(drop=True)

    return train_failed_df, test_failed_df


def build_nonfailed_failure_split(
        performance_split_df: pd.DataFrame,
        feature_cols: list[str],
) -> pd.DataFrame:
    missing_cols = [col for col in feature_cols if col not in performance_split_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    result = (
        performance_split_df[feature_cols]
        .sort_values("config_id")
        .drop_duplicates(subset="config_id", keep="first")
        .reset_index(drop=True)
    )

    result["failed"] = 0

    return result


def combine_failure_split(
        nonfailed_df: pd.DataFrame,
        failed_df: pd.DataFrame,
) -> pd.DataFrame:
    result = pd.concat([nonfailed_df, failed_df], ignore_index=True)
    result = result.sort_values("config_id").reset_index(drop=True)

    return result


def prepare_divergence_predictor_data(
        full_df: pd.DataFrame,
        feature_cols: list[str],
        train_output_csv: str,
        test_output_csv: str,
        test_size: float = 0.2,
        random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_failure_df = build_full_failure_dataset(full_df, feature_cols)

    train_df, test_df = train_test_split(
        full_failure_df,
        test_size=test_size,
        stratify=full_failure_df["failed"],
        random_state=random_state,
    )

    train_df.to_csv(train_output_csv, index=False)
    test_df.to_csv(test_output_csv, index=False)

    return train_df, test_df


def main() -> None:
    full_csv = "data.csv"
    flops_col = "Total compute(GLOPs)"

    divergence_train_csv = "divergence_surrogate/splits/train.csv"
    divergence_test_csv = "divergence_surrogate/splits/test.csv"

    full_df = pd.read_csv(full_csv)

    nonfailed_last_epoch_df = keep_only_nonfailed_last_epoch(full_df)

    prepare_performance_predictor_data(
        full_df=full_df,
        df=nonfailed_last_epoch_df,
        flops_col=flops_col,
        num_bins=4,
        random_state=42,
    )

    prepare_divergence_predictor_data(
        full_df=full_df,
        feature_cols=FEATURE_COLS,
        train_output_csv=divergence_train_csv,
        test_output_csv=divergence_test_csv,
        test_size=0.2,
        random_state=42,
    )


if __name__ == "__main__":
    main()
