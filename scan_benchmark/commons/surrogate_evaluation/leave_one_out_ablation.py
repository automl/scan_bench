from copy import deepcopy

from scan_benchmark.commons.train.train_base import run_benchmark
from scan_benchmark.vlm.performance_surrogate.data import VLMSurrogateDataset
from scan_benchmark.vlm.performance_surrogate.train.train import parse_args


if __name__ == "__main__":
    folds = [1, 2, 3, 4, 5]
    default_features = VLMSurrogateDataset.DEFAULT_FEATURES

    base_args = parse_args()

    base_args.additional_runs_path = None
    base_args.include_intermediate_points = True
    base_args.eval_on_intermediate_points = True

    for removed_feature in default_features:
        for fold in folds:
            ablation_args = deepcopy(base_args)

            ablation_args.fold = fold
            ablation_args.train_csv = (
                f"../../vlm/performance_surrogate/splits/train_fold_{fold}.csv"
            )
            ablation_args.test_csv = (
                f"../../vlm/performance_surrogate/splits/test_fold_{fold}.csv"
            )

            ablation_args.features = [
                feature
                for feature in default_features
                if feature != removed_feature
            ]

            ablation_args.out_dir = (
                f"ablation_results/"
                f"without_{removed_feature}/"
                f"fold_{fold}"
            )

            print(
                f"\nRemoving feature '{removed_feature}', "
                f"running fold {fold}"
            )

            run_benchmark(
                args=ablation_args,
                dataset_cls=VLMSurrogateDataset,
                supports_intermediate_points=True,
            )
