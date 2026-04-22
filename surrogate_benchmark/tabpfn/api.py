from importlib.resources import files

from surrogate_benchmark.base_performance_benchmark import BasePerformanceBenchmark
from surrogate_benchmark.tabpfn.config import TabPFNConfig, TabPFNTarget
from surrogate_benchmark.tabpfn.performance_surrogate.data import TabPFNSurrogateDataset


class TabPFNBenchmark(BasePerformanceBenchmark):
    TARGET_ENUM = TabPFNTarget

    def __init__(self, targets=None, device="auto"):
        targets = self._normalize_targets(targets)

        train_path = files("surrogate_benchmark.tabpfn.performance_surrogate").joinpath("splits/train_small.csv")
        test_path = files("surrogate_benchmark.tabpfn.performance_surrogate").joinpath("splits/test_small.csv")

        dataset = TabPFNSurrogateDataset(
            train_csv_path=str(train_path),
            test_csv_path=str(test_path),
            targets=targets,
            seed=42,
        )
        super().__init__(surrogate_dataset=dataset, device=device)

    def query(self, config: TabPFNConfig) -> dict:
        return {"predictions": self._predict_performance(config)}

    def query_many(self, configs: list[TabPFNConfig]) -> list[dict]:
        preds = self._predict_performance_many(configs)
        return [{"predictions": p} for p in preds]


if __name__ == "__main__":
    tabpfn_bench = TabPFNBenchmark()

    config = TabPFNConfig(
        total_cells=1_000_000,
        effective_batch_size=64,
        lr=0.01,
        max_features=8,
        embedding_size=32,
        num_layers=4,
        num_datapoints_max=400,
        weight_decay=0.0,
    )

    result = tabpfn_bench.query_many([config])
    print(result)
