import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_log_bins(
        df: pd.DataFrame,
        flops_col: str,
        num_bins: int = 3,
):
    x = df[flops_col].values
    log_x = np.log(x)

    bins = np.linspace(log_x.min(), log_x.max(), num_bins + 1)
    labels = [f"bin_{i}" for i in range(num_bins)]

    df = df.copy()
    df["flops_bin"] = pd.cut(
        log_x,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    return df, bins, labels


def split_each_bin(
        df: pd.DataFrame,
        labels: list[str],
        test_size: float = 0.2,
        random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []

    for bin_label in labels:
        bin_df = df[df["flops_bin"] == bin_label].copy()

        train_bin, test_bin = train_test_split(
            bin_df,
            test_size=test_size,
            random_state=random_state,
        )

        train_parts.append(train_bin)
        test_parts.append(test_bin)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, test_df


def downsample_preserving_bins(
        df: pd.DataFrame,
        labels: list[str],
        n_samples: int,
        random_state: int = 42,
) -> pd.DataFrame:
    if len(df) <= n_samples:
        return df.copy()

    rng = np.random.default_rng(random_state)

    counts = df["flops_bin"].value_counts().reindex(labels, fill_value=0)
    proportions = counts / counts.sum()

    target_counts = (proportions * n_samples).astype(int)

    remainder = n_samples - target_counts.sum()
    if remainder > 0:
        frac_parts = (proportions * n_samples) - target_counts
        extra_bins = frac_parts.sort_values(ascending=False).index[:remainder]
        for b in extra_bins:
            target_counts[b] += 1

    sampled_parts = []
    for bin_label in labels:
        bin_df = df[df["flops_bin"] == bin_label]
        k = min(target_counts[bin_label], len(bin_df))

        if k > 0:
            sampled_parts.append(
                bin_df.sample(n=k, random_state=random_state)
            )

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return sampled_df


def print_bin_counts(df: pd.DataFrame, col: str = "flops_bin") -> None:
    counts = df[col].value_counts().sort_index()
    print("\nCounts per bin:")
    print(counts)


def plot_gflops_distribution(df, flops_col, bins):
    x = df[flops_col]

    plt.figure(figsize=(8, 2))
    plt.scatter(x, [0] * len(x), alpha=0.6)

    for b in np.exp(bins):
        plt.axvline(b, linestyle="--")

    plt.xscale("log")
    plt.yticks([])
    plt.xlabel("FLOPs")
    plt.title("Log-spaced bins")
    plt.show()


def prepare_splits(
        df: pd.DataFrame,
        flops_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        small_scale: bool = False,
        small_train_size: int = 1000,
        small_test_size: int = 500,
):
    df, bins, labels = create_log_bins(df, flops_col=flops_col)

    print_bin_counts(df)
    plot_gflops_distribution(df, flops_col=flops_col, bins=bins)

    train_df, test_df = split_each_bin(
        df=df,
        labels=labels,
        test_size=test_size,
        random_state=random_state,
    )

    if small_scale:
        train_df = downsample_preserving_bins(
            train_df,
            labels=labels,
            n_samples=small_train_size,
            random_state=random_state,
        )
        test_df = downsample_preserving_bins(
            test_df,
            labels=labels,
            n_samples=small_test_size,
            random_state=random_state,
        )

    train_df = train_df.drop(columns=["flops_bin"])
    test_df = test_df.drop(columns=["flops_bin"])

    return train_df, test_df


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    flops_col = "total_flops"

    small_scale = True

    train_df, test_df = prepare_splits(
        df=df,
        flops_col=flops_col,
        test_size=0.2,
        random_state=42,
        small_scale=small_scale,
        small_train_size=2000,
        small_test_size=500,
    )

    suffix = "_small" if small_scale else ""

    train_df.to_csv(f"performance_surrogate/splits/train{suffix}.csv", index=False)
    test_df.to_csv(f"performance_surrogate/splits/test{suffix}.csv", index=False)
