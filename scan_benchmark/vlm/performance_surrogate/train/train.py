from scan_benchmark.commons.train.args import get_base_parser, finalize_args
from scan_benchmark.commons.train.train_base import run_benchmark
from scan_benchmark.vlm.performance_surrogate.data import VLMSurrogateDataset


def parse_args():
    p = get_base_parser()

    p.add_argument("--include_intermediate_points", action="store_true")
    p.add_argument("--eval_on_intermediate_points", action="store_true")

    args = p.parse_args()

    defaults = {
        "train_csv": "scan_benchmark/vlm/performance_surrogate/splits/train_fold_1.csv",
        "test_csv": "scan_benchmark/vlm/performance_surrogate/splits/test_fold_1.csv",
        "labels": ["val_loss"],
        "additional_runs_path": "scan_benchmark/vlm/large_runs.csv",
    }

    return finalize_args(args, defaults)


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        args=args,
        dataset_cls=VLMSurrogateDataset,
        supports_intermediate_points=True,
        model_family="vlm",
    )
