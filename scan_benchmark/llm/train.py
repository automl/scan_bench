from scan_benchmark.commons.train.args import get_base_parser, finalize_args
from scan_benchmark.commons.train.train_base import run_benchmark
from scan_benchmark.llm.data import LLMSurrogateDataset


def parse_args():
    p = get_base_parser()

    p.add_argument("--include_intermediate_points", action="store_true")
    p.add_argument("--eval_on_intermediate_points", action="store_true")

    args = p.parse_args()

    defaults = {
        "train_csv": "scan_benchmark/llm/splits/train.csv",
        "test_csv": "scan_benchmark/llm/splits/test.csv",
        "labels": ["test_loss"],
    }

    return finalize_args(args, defaults)


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        args=args,
        dataset_cls=LLMSurrogateDataset,
        supports_intermediate_points=True,
    )
