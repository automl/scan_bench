from importlib.resources import files
from pprint import pprint

from scan_benchmark.base_performance_benchmark import BasePerformanceBenchmark
from scan_benchmark.tabpfn.config import TabPFNConfig, TabPFNTarget
from scan_benchmark.tabpfn.performance_surrogate.data import TabPFNSurrogateDataset


class TabPFNBenchmark(BasePerformanceBenchmark):
    TARGET_ENUM = TabPFNTarget

    def __init__(self, targets=None, device="auto"):
        targets = self._normalize_targets(targets)
        if TabPFNTarget.FLOPS.value not in targets:
            targets.append(TabPFNTarget.FLOPS.value)

        train_path = files("scan_benchmark.tabpfn.performance_surrogate").joinpath("splits/train.csv")
        test_path = files("scan_benchmark.tabpfn.performance_surrogate").joinpath("splits/test.csv")

        dataset = TabPFNSurrogateDataset(
            train_csv_path=str(train_path),
            test_csv_path=str(test_path),
            targets=targets,
            seed=42,
        )
        super().__init__(surrogate_dataset=dataset, device=device)

    def query(self, config: TabPFNConfig) -> dict:
        preds = self._predict_performance(config)

        flops_val = preds.get(TabPFNTarget.FLOPS.value)

        if isinstance(flops_val, dict):
            flops_val = flops_val.get("mean")

        return {
            "predictions": {
                k: v for k, v in preds.items()
                if k != TabPFNTarget.FLOPS.value
            },
            "model_stats": {
                "flops": flops_val
            },
        }

    def query_many(self, configs: list[TabPFNConfig]) -> list[dict]:
        preds_list = self._predict_performance_many(configs)

        results = []
        for preds in preds_list:
            flops_val = preds.get(TabPFNTarget.FLOPS.value)

            if isinstance(flops_val, dict):
                flops_val = flops_val.get("mean")

            results.append({
                "predictions": {
                    k: v for k, v in preds.items()
                    if k != TabPFNTarget.FLOPS.value
                },
                "model_stats": {
                    "flops": flops_val
                },
            })

        return results


if __name__ == "__main__":
    tabpfn_bench = TabPFNBenchmark()

    config = TabPFNConfig(
        total_cells=2_000_000,
        effective_batch_size=64,
        lr=0.01,
        max_features=32,
        embedding_size=32,
        num_layers=4,
        num_datapoints_max=400,
    )

    result = tabpfn_bench.query_many([config])
    pprint(result)
