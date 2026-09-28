import numpy as np
import pandas as pd


def build_failure_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")

    df["epoch_failed"] = (
        df["epoch_diverged"].astype(bool)
        | np.isinf(df["train_loss"])
    )

    fail_df = (
        df.groupby("config_id", as_index=False)["epoch_failed"]
        .any()
        .rename(columns={"epoch_failed": "failed"})
    )

    fail_df["failed"] = fail_df["failed"].astype(int)
    return fail_df


if __name__ == "__main__":
    df = pd.read_csv("data.csv")

    fail_df = build_failure_labels(df)

    df = df.merge(fail_df, on="config_id", how="left")
    df_nonfailed = df[df["failed"] == 0].copy()

    a100_df = df_nonfailed[
        df_nonfailed["train_gpu_name"].str.contains(
            "NVIDIA A100-SXM4-40GB",
            case=False,
            na=False
        )
    ].copy()

    a100_df["epoch_gpu_time_seconds"] = (
            a100_df["world_size"] * a100_df["train_duration(s)"]
    )

    config_times = (
        a100_df
        .groupby("config_id", as_index=False)
        .agg(
            total_gpu_time_seconds=("epoch_gpu_time_seconds", "sum"),
            total_gpu_time_hours=("epoch_gpu_time_seconds", lambda x: x.sum() / 3600),
            num_epochs=("epoch", "nunique"),
            num_rows=("epoch_gpu_time_seconds", "size"),
        )
    )

    max_config = config_times.loc[
        config_times["total_gpu_time_seconds"].idxmax()
    ]

    min_config = config_times.loc[
        config_times["total_gpu_time_seconds"].idxmin()
    ]

    print("Config with highest total GPU time:")
    print(max_config)

    print("\nConfig with lowest total GPU time:")
    print(min_config)
