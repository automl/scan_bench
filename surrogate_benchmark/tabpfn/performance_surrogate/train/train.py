from surrogate_benchmark.commons.train.args import get_base_parser, finalize_args
from surrogate_benchmark.commons.train.train_base import run_benchmark
from surrogate_benchmark.tabpfn.performance_surrogate.data import TabPFNSurrogateDataset


def parse_args():
    p = get_base_parser()
    args = p.parse_args()

    defaults = {
        "train_csv": "../splits/train_small.csv",
        "test_csv": "../splits/test_small.csv",
        "labels": ["val/val_loss"],
    }

    return finalize_args(args, defaults)


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        args=args,
        dataset_cls=TabPFNSurrogateDataset,
        supports_intermediate_points=False,
    )
