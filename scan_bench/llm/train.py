import copy
from pathlib import Path

from scan_bench.commons.train.args import get_base_parser, finalize_args
from scan_bench.commons.train.train_base import build_out_path, run_benchmark
from scan_bench.llm.data import LLMSurrogateDataset


DEFAULT_SPLIT_DIR = Path("scan_bench/llm/splits")


def parse_args():
    p = get_base_parser()

    p.add_argument("--include_intermediate_points", action="store_true")
    p.add_argument("--eval_on_intermediate_points", action="store_true")

    args = p.parse_args()

    defaults = {
        "train_csv": "scan_bench/llm/splits/train_split_1.csv",
        "test_csv": "scan_bench/llm/splits/test_split_1.csv",
        "labels": ["test_loss"],
    }

    return finalize_args(args, defaults)


def run_one_fold(args, fold: int):
    fold_args = copy.copy(args)
    fold_args.fold = int(fold)
    fold_args.model_family = "llm"
    fold_args.use_fold_in_path = True
    fold_args.train_csv = str(DEFAULT_SPLIT_DIR / f"train_split_{fold}.csv")
    fold_args.test_csv = str(DEFAULT_SPLIT_DIR / f"test_split_{fold}.csv")

    if args.out_dir is not None:
        fold_args.out_dir = str(Path(args.out_dir) / f"split_{fold}")
    else:
        fold_args.out_dir = str(build_out_path(fold_args) / f"split_{fold}")

    return run_benchmark(
        args=fold_args,
        dataset_cls=LLMSurrogateDataset,
        supports_intermediate_points=True,
        model_family="llm",
    )


if __name__ == "__main__":
    args = parse_args()
    if args.folds:
        for fold in args.folds:
            run_one_fold(args, fold)
    else:
        run_benchmark(
            args=args,
            dataset_cls=LLMSurrogateDataset,
            supports_intermediate_points=True,
            model_family="llm",
        )