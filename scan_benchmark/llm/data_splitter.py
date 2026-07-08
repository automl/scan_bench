import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold


FEATURE_COLUMNS = [
    "config_id",
    "d_model",
    "n_layers",
    "n_heads",
    "weight_decay",
    "beta1",
    "beta2",
    # "warmup_steps",
    "cooldown_steps",
    "lr",
    "global_batch_size",
    "final_step",
    "total_compute",
    "n_data",
    "n_param",
    # "micro_batch_size",
    "current_lr",
    "tokens_so_far",
    # "train_loss_so_far",
    "flops_so_far",
    "eval_step",
]

LOSS_TARGET_COLUMNS = [
    "valid_loss",
    "test_loss",
]

# DOWNSTREAM_TARGET_RENAMES = {
#     "downstream_hellaswag": "hellaswag_acc",
#     "downstream_arc_challenge": "arc_challenge_acc",
#     "downstream_arc_easy": "arc_easy_acc",
#     "downstream_copa": "copa_acc",
#     "downstream_openbook_qa": "openbookqa_acc",
#     "downstream_piqa": "piqa_acc",
# }


def normalize_config_id(value) -> str:
    if isinstance(value, str) and value.startswith("config_"):
        return value
    return f"config_{int(float(value))}"


def save_bin_boundaries(bins, labels=None, filepath="bin_boundaries.txt"):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

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


def print_filled_values(
        before: pd.Series,
        after: pd.Series,
        column: str,
) -> None:
    before_missing = ~np.isfinite(pd.to_numeric(before, errors="coerce"))
    after_present = np.isfinite(pd.to_numeric(after, errors="coerce"))
    filled_count = int((before_missing & after_present).sum())
    if filled_count:
        print(f"Filled {filled_count} missing {column} values.")


def print_removed_rows(
        original_df: pd.DataFrame,
        filtered_df: pd.DataFrame,
        reason: str,
) -> None:
    removed_rows = len(original_df) - len(filtered_df)
    if removed_rows == 0:
        return

    original_configs = set(original_df["config_id"].unique())
    filtered_configs = set(filtered_df["config_id"].unique())
    removed_configs = sorted(original_configs - filtered_configs)

    print(f"\nRemoved {removed_rows} rows while {reason}.")
    if removed_configs:
        print(
            f"WARNING: removed all rows for {len(removed_configs)} configs: "
            f"{removed_configs}"
        )
    else:
        print("No configs were fully removed.")


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def read_summary_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, ["config_id"])

    df = df.copy()
    df["config_id"] = df["config_id"].map(normalize_config_id)
    return df


def compute_progress(
        eval_step: pd.Series,
        final_step: pd.Series,
        step_fraction: pd.Series | None = None,
) -> pd.Series:
    eval_step = pd.to_numeric(eval_step, errors="coerce")
    final_step = pd.to_numeric(final_step, errors="coerce")

    progress = pd.Series(np.nan, index=eval_step.index, dtype=float)
    valid_final_step = final_step > 0
    progress.loc[valid_final_step] = (
        eval_step.loc[valid_final_step] / final_step.loc[valid_final_step]
    )

    if step_fraction is not None:
        fallback_progress = pd.to_numeric(step_fraction, errors="coerce")
        progress = progress.fillna(fallback_progress)

    return progress.clip(lower=0.0, upper=1.0)


def compute_current_lr(
    eval_step: pd.Series,
    final_step: pd.Series,
    lr: pd.Series,
    warmup_steps: pd.Series,
    cooldown_steps: pd.Series,
    lr_end: pd.Series,
) -> pd.Series:
    t = pd.to_numeric(eval_step, errors="coerce").to_numpy(dtype=float)
    final = pd.to_numeric(final_step, errors="coerce").to_numpy(dtype=float)
    initial_lr = pd.to_numeric(lr, errors="coerce").to_numpy(dtype=float)
    warmup = pd.to_numeric(warmup_steps, errors="coerce").to_numpy(dtype=float)
    cooldown = pd.to_numeric(cooldown_steps, errors="coerce").to_numpy(dtype=float)
    end_lr = pd.to_numeric(lr_end, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    warmup_abs = warmup * final
    cooldown_abs = cooldown * final
    cooldown_start = final - cooldown_abs

    current_lr = np.full_like(initial_lr, np.nan, dtype=float)
    valid = np.isfinite(t) & np.isfinite(final) & np.isfinite(initial_lr)
    warmup_mask = valid & (t <= warmup_abs)
    plateau_mask = valid & (~warmup_mask) & (t <= cooldown_start)
    cooldown_zero_mask = valid & (~warmup_mask) & (~plateau_mask) & (cooldown_abs <= 0)
    cooldown_mask = valid & (~warmup_mask) & (~plateau_mask) & (~cooldown_zero_mask)

    current_lr[warmup_mask] = (
        initial_lr[warmup_mask]
        * t[warmup_mask]
        / np.maximum(warmup_abs[warmup_mask], 1.0)
    )
    current_lr[plateau_mask] = initial_lr[plateau_mask]
    current_lr[cooldown_zero_mask] = end_lr[cooldown_zero_mask]

    progress = np.zeros_like(t)
    progress[cooldown_mask] = (
        (t[cooldown_mask] - cooldown_start[cooldown_mask])
        / cooldown_abs[cooldown_mask]
    )
    progress = np.clip(progress, 0.0, 1.0)
    current_lr[cooldown_mask] = (
        initial_lr[cooldown_mask]
        + (end_lr[cooldown_mask] - initial_lr[cooldown_mask])
        * progress[cooldown_mask]
    )

    return pd.Series(current_lr, index=eval_step.index)


def fill_scaled_progress_column(
        df: pd.DataFrame,
        column: str,
        budget_column: str,
        progress: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    budget = pd.to_numeric(df[budget_column], errors="coerce")
    replacement = budget * progress

    missing_values = ~np.isfinite(values)
    usable_replacements = np.isfinite(replacement)
    values.loc[missing_values & usable_replacements] = replacement.loc[
        missing_values & usable_replacements
    ]
    return values


def compact_llm_dataframe(summary_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        summary_df,
        [
            "config_id",
            "d_model",
            "n_layers",
            "n_heads",
            "weight_decay",
            "beta1",
            "beta2",
            "warmup_steps",
            "cooldown_steps",
            "lr",
            "current_lr",
            "micro_batch_size",
            "grad_accumulation_steps",
            "recommended_num_gpus",
            # "resume_step",
            "total_flops",
            "n_tokens",
            "total_params",
            "tokens_so_far",
            # "train_loss_so_far",
            "flops_so_far",
            "eval_step",
        ]
        + LOSS_TARGET_COLUMNS,
    )

    global_batch_size = (
        summary_df["micro_batch_size"]
        * summary_df["grad_accumulation_steps"]
        * summary_df["recommended_num_gpus"]
    )
    final_step = summary_df["resume_step"].copy()
    fallback_final_step = summary_df.groupby("config_id")["eval_step"].transform("max")
    final_step = final_step.fillna(fallback_final_step)

    if "current_lr" in summary_df.columns:
        current_lr = summary_df["current_lr"].copy()
    else:
        current_lr = pd.Series(np.nan, index=summary_df.index)

    computed_current_lr = compute_current_lr(
        eval_step=summary_df["eval_step"],
        final_step=final_step,
        lr=summary_df["lr"],
        warmup_steps=summary_df["warmup_steps"],
        cooldown_steps=summary_df["cooldown_steps"],
        lr_end=summary_df.get("lr_end", pd.Series(0.0, index=summary_df.index)),
    )
    repaired_current_lr = current_lr.fillna(computed_current_lr)
    print_filled_values(current_lr, repaired_current_lr, "current_lr")
    current_lr = repaired_current_lr

    df = pd.DataFrame(
        {
            "config_id": summary_df["config_id"],
            "d_model": summary_df["d_model"],
            "n_layers": summary_df["n_layers"],
            "n_heads": summary_df["n_heads"],
            "weight_decay": summary_df["weight_decay"],
            "beta1": summary_df["beta1"],
            "beta2": summary_df["beta2"],
            # "warmup_steps": summary_df["warmup_steps"],
            "cooldown_steps": summary_df["cooldown_steps"],
            "lr": summary_df["lr"],
            "global_batch_size": global_batch_size,
            "final_step": final_step,
            "total_compute": summary_df["total_flops"],
            "n_data": summary_df["n_tokens"],
            "n_param": summary_df["total_params"],
            "micro_batch_size": summary_df["micro_batch_size"],
            "current_lr": current_lr,
            "tokens_so_far": summary_df["tokens_so_far"],
            # "train_loss_so_far": summary_df["train_loss_so_far"],
            "flops_so_far": summary_df["flops_so_far"],
            "eval_step": summary_df["eval_step"],
        }
    )

    progress = compute_progress(
        eval_step=df["eval_step"],
        final_step=df["final_step"],
        step_fraction=summary_df.get("step_fraction"),
    )
    for col, budget_col in [
        ("tokens_so_far", "n_data"),
        ("flops_so_far", "total_compute"),
    ]:
        original_values = df[col].copy()
        repaired_values = fill_scaled_progress_column(
            df=df,
            column=col,
            budget_column=budget_col,
            progress=progress,
        )
        print_filled_values(original_values, repaired_values, col)
        df[col] = repaired_values

    for col in LOSS_TARGET_COLUMNS:
        df[col] = summary_df[col]

    # for source_col, target_col in DOWNSTREAM_TARGET_RENAMES.items():
    #     if source_col in summary_df.columns:
    #         df[target_col] = summary_df[source_col]

    # target_columns = LOSS_TARGET_COLUMNS + list(DOWNSTREAM_TARGET_RENAMES.values())
    target_columns = [col for col in LOSS_TARGET_COLUMNS if col in df.columns]

    compact_df = (
        df.sort_values(["config_id", "eval_step"])
        .dropna(subset=FEATURE_COLUMNS + target_columns)
        .reset_index(drop=True)
    )
    print_removed_rows(
        original_df=df,
        filtered_df=compact_df,
        reason="dropping rows with missing features or targets",
    )
    return compact_df


def get_last_eval_per_config(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["config_id", "eval_step"])
        .drop_duplicates(subset="config_id", keep="last")
        .reset_index(drop=True)
    )


def enrich_with_all_eval_steps(
        full_df: pd.DataFrame,
        selected_configs_df: pd.DataFrame,
) -> pd.DataFrame:
    selected_config_ids = selected_configs_df["config_id"].unique()
    enriched_df = full_df[full_df["config_id"].isin(selected_config_ids)].copy()

    if "flops_bin" in selected_configs_df.columns:
        enriched_df = enriched_df.merge(
            selected_configs_df[["config_id", "flops_bin"]].drop_duplicates(),
            on="config_id",
            how="left",
        )

    return (
        enriched_df
        .sort_values(["config_id", "eval_step"])
        .reset_index(drop=True)
    )


def prepare_performance_predictor_data(
        full_df: pd.DataFrame,
        df: pd.DataFrame,
        flops_col: str,
        output_dir: str | Path,
        num_bins: int = 3,
        num_folds: int = 5,
        random_state: int = 42,
        make_plot: bool = True,
):
    x = pd.to_numeric(df[flops_col], errors="coerce").to_numpy(dtype=float)
    log_x = np.log10(np.clip(x, np.finfo(float).tiny, None))

    bins = np.quantile(log_x, np.linspace(0, 1, num_bins + 1))
    bins = np.unique(bins)
    labels = [f"bin_{i}" for i in range(len(bins) - 1)]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_bin_boundaries(bins, labels, filepath=output_dir / "bin_boundaries.txt")

    df = df.copy()
    df["flops_bin"] = pd.cut(
        log_x,
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(str)

    # if make_plot:
    #     plot_gflops_distribution(df, flops_col=flops_col, bins=bins)

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

        train_df = enrich_with_all_eval_steps(full_df, train_last_df)
        test_df = enrich_with_all_eval_steps(full_df, test_last_df)

        train_output_csv = output_dir / f"train_fold_{fold_idx}.csv"
        test_output_csv = output_dir / f"test_fold_{fold_idx}.csv"

        train_df.to_csv(train_output_csv, index=False)
        test_df.to_csv(test_output_csv, index=False)

        if fold_idx == 1:
            train_df.to_csv(output_dir / "train.csv", index=False)
            test_df.to_csv(output_dir / "test.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="scan_benchmark/llm/data/configs_intermediate_summary.csv",
    )
    parser.add_argument("--output_dir", default="scan_benchmark/llm/splits")
    parser.add_argument("--flops_col", default="total_compute")
    parser.add_argument("--num_bins", type=int, default=3)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_df = read_summary_csv(args.input_csv)
    compact_df = compact_llm_dataframe(summary_df)
    last_eval_df = get_last_eval_per_config(compact_df)

    prepare_performance_predictor_data(
        full_df=compact_df,
        df=last_eval_df,
        flops_col=args.flops_col,
        output_dir=args.output_dir,
        num_bins=args.num_bins,
        num_folds=args.num_folds,
        random_state=args.seed,
        make_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
