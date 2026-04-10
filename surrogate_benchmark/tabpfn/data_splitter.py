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

    for bin in labels:
        bin_df = df[df["flops_bin"] == bin].copy()

        train_bin, test_bin = train_test_split(
            bin_df,
            test_size=test_size,
            random_state=random_state,
        )

        train_parts.append(train_bin)
        test_parts.append(test_bin)

    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts)

    train_df = train_df.drop(columns=["flops_bin"])
    test_df = test_df.drop(columns=["flops_bin"])

    return train_df, test_df


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


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    flops_col = "total_flops"
    df, bins, labels = create_log_bins(df, flops_col=flops_col)

    print_bin_counts(df)

    plot_gflops_distribution(df, flops_col=flops_col, bins=bins)

    train_df, test_df = split_each_bin(
        df=df,
        labels=labels,
        test_size=0.2,
        random_state=42,
    )

    train_df.to_csv("performance_surrogate/splits/train.csv", index=False)
    test_df.to_csv("performance_surrogate/splits/test.csv", index=False)
