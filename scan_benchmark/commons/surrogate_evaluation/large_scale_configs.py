from pathlib import Path

import pandas as pd

from scan_benchmark.commons.train.train_base import run_benchmark
from scan_benchmark.vlm.performance_surrogate.data import VLMSurrogateDataset
from scan_benchmark.vlm.performance_surrogate.train.train import parse_args

if __name__ == "__main__":
    args = parse_args()
    # split_dir = Path("../../vlm/performance_surrogate/splits")
    #
    # train_df = pd.concat([
    #     pd.read_csv(split_dir / "train_fold_1.csv"),
    #     pd.read_csv(split_dir / "test_fold_1.csv"),
    # ], ignore_index=True)
    #
    # merged_train_path = "train_all_fold_1.csv"
    # train_df.to_csv(merged_train_path, index=False)
    #
    # test_path = "../../vlm/large_runs.csv"
    # filtered_test_path = "large_runs_filtered.csv"
    #
    # df = pd.read_csv(test_path)
    #
    # ignored_configs = df.loc[
    #     (df["epoch"] == 10) & (df["val_loss"] > 1),
    #     "config_id",
    # ].unique()
    #
    # filtered_df = df[~df["config_id"].isin(ignored_configs)].copy()
    # filtered_df.to_csv(filtered_test_path, index=False)

    args.train_csv = "../../vlm/performance_surrogate/splits/train_fold_1.csv"
    args.test_csv = "../../vlm/performance_surrogate/splits/test_fold_1.csv"
    args.additional_runs_path = "../../vlm/large_runs.csv"
    args.include_intermediate_points = True
    args.eval_on_intermediate_points = True
    args.out_dir = str(Path("results") / "fold_1")

    run_benchmark(
        args=args,
        dataset_cls=VLMSurrogateDataset,
        supports_intermediate_points=True,
    )
