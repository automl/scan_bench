import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_log_bins(
        df: pd.DataFrame,
        flops_col: str,
        num_bins: int = 5,
):
    x = df[flops_col].values
    log_x = np.log10(x)

    bins = np.linspace(log_x.min(), log_x.max(), num_bins + 1)

    def print_bin_boundaries(bins):
        print("\nBin boundaries:")

        for i in range(len(bins) - 1):
            low_log = bins[i]
            high_log = bins[i + 1]

            low = 10 ** low_log
            high = 10 ** high_log

            print(
                f"bin_{i}: "
                f"log10 [{low_log:.2f}, {high_log:.2f}]  |  "
                f"FLOPs [{low:.2e}, {high:.2e}]"
            )

    print_bin_boundaries(bins)
    labels = [f"bin_{i}" for i in range(num_bins)]

    df = df.copy()
    df["flops_bin"] = pd.cut(
        log_x,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    return df, bins, labels


def print_bin_counts(df: pd.DataFrame, col: str = "flops_bin") -> None:
    counts = df[col].value_counts().sort_index()
    print("\nCounts per bin:")
    print(counts)


def plot_gflops_distribution(df, flops_col, bins):
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


def prepare_splits(
        df: pd.DataFrame,
        flops_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
):
    df, bins, labels = create_log_bins(df, flops_col=flops_col)

    save_bin_boundaries(bins)
    print_bin_counts(df)
    plot_gflops_distribution(df, flops_col=flops_col, bins=bins)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["flops_bin"],
        shuffle=True,
    )

    print("\n--- Train split ---")
    print_bin_counts(train_df)

    print("\n--- Test split ---")
    print_bin_counts(test_df)

    return train_df, test_df

def save_bin_boundaries(bins, filepath="performance_surrogate/splits/bin_boundaries.txt"):
    with open(filepath, "w") as f:
        f.write("Bin boundaries:\n\n")

        for i in range(len(bins) - 1):
            low_log = bins[i]
            high_log = bins[i + 1]

            low = 10 ** low_log
            high = 10 ** high_log

            line = (
                f"bin_{i}: "
                f"log10 [{low_log:.2f}, {high_log:.2f}]  |  "
                f"FLOPs [{low:.2e}, {high:.2e}]\n"
            )

            f.write(line)

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    flops_col = "total_flops"

    train_df, test_df = prepare_splits(
        df=df,
        flops_col=flops_col,
        test_size=0.2,
        random_state=42,
    )

    train_df.to_csv("performance_surrogate/splits/train.csv", index=False)
    test_df.to_csv("performance_surrogate/splits/test.csv", index=False)
