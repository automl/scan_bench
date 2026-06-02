import json
from pathlib import Path

import numpy as np
import pandas as pd

from scan_benchmark.commons.train.train_base import run_benchmark
from scan_benchmark.vlm.performance_surrogate.data import VLMSurrogateDataset
from scan_benchmark.vlm.performance_surrogate.train.train import parse_args

if __name__ == "__main__":
    train_split = "../../vlm/performance_surrogate/splits/train.csv"
    test_split = "../../vlm/performance_surrogate/splits/test.csv"

    train_df = pd.read_csv(train_split)
    unique_configs = train_df["config_id"].unique()

    output_dir = Path("variance_subsets")
    output_dir.mkdir(exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    args = parse_args()

    for seed in range(10):
        selected_configs = pd.Series(unique_configs).sample(
            frac=0.8,
            random_state=seed
        )

        train_subset = train_df[
            train_df["config_id"].isin(selected_configs)
        ].copy()

        subset_path = output_dir / f"train_seed_{seed}.csv"
        train_subset.to_csv(subset_path, index=False)

        args.train_csv = str(subset_path)
        args.test_csv = test_split
        args.include_intermediate_points = True
        args.eval_on_intermediate_points = True
        args.out_dir = str(results_dir / f"seed_{seed}")

        run_benchmark(
            args=args,
            dataset_cls=VLMSurrogateDataset,
            supports_intermediate_points=True,
        )

    results_root = Path("results")

    metrics = {}

    for seed_dir in sorted(results_root.glob("seed_*")):
        result_file = (seed_dir / "val_loss.json")

        with open(result_file) as f:
            data = json.load(f)

        result = list(data.values())[-1]

        if not metrics:
            metrics = {metric: [] for metric in result}

        for metric, value in result.items():
            metrics[metric].append(value)

    for metric, values in metrics.items():
        values = np.array(values)

        print(
            f"{metric:10s}: "
            f"{values.mean():.4f} ± {values.std():.4f} "
        )
