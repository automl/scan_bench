import json
from importlib.resources import files
from pprint import pprint

from scan_benchmark.base_performance_benchmark import BasePerformanceBenchmark, PerformancePredictorType, is_tabpfn
from scan_benchmark.tabpfn.config import TabPFNConfig, TabPFNTarget
from scan_benchmark.tabpfn.performance_surrogate.data import TabPFNSurrogateDataset


class TabPFNBenchmark(BasePerformanceBenchmark):
    TARGET_ENUM = TabPFNTarget

    def __init__(self, target=None, predictor_type: PerformancePredictorType = PerformancePredictorType.TABPFN,
                 autogluon_model_path: str | None = None, device="auto"):

        train_path = files("scan_benchmark.tabpfn").joinpath("data.csv")

        dataset = TabPFNSurrogateDataset(
            train_csv_path=str(train_path),
            targets=target,
            seed=42,
        )
        self.default_exponential = [
            "effective_batch_size",
            "total_cells",
            "num_layers",
            "max_features",
            "num_datapoints_max",
        ]

        if is_tabpfn(predictor_type):
            saved_model_path = None

        elif predictor_type == PerformancePredictorType.AUTOGLUON:
            saved_model_path = (
                autogluon_model_path
                if autogluon_model_path is not None
                else files("scan_benchmark.tabpfn.performance_surrogate").joinpath(
                    "saved_models",
                    "predictors",
                    predictor_type.value,
                    f"seed_42",
                    "auto"
                )
            )
        else:
            saved_model_path = files("scan_benchmark.tabpfn.performance_surrogate").joinpath(
                "saved_models",
                "predictors",
                predictor_type.value,
                f"seed_42",
            )

        super().__init__(
            target=target,
            surrogate_dataset=dataset,
            model_path=saved_model_path,
            predictor_type=predictor_type,
            device=device,
        )

    def query(self, config: TabPFNConfig) -> dict:
        preds = self._predict_performance(config)

        return {
            "predictions": {
                k: v for k, v in preds.items()
            },
            "model_stats": {
                "flops": self.flops(config),
                "params": self.model_params(config),
            },
        }

    def query_many(self, configs: list[TabPFNConfig]) -> list[dict]:
        preds_list = self._predict_performance_many(configs)

        results = []
        for config, preds in zip(configs, preds_list):
            n_params = self.model_params(config)
            flops = self.flops(config)

            results.append({
                "predictions": {
                    k: v for k, v in preds.items()
                },
                "model_stats": {
                    "flops": flops,
                    "n_parameters": n_params,
                },
            })

        return results

    def model_params(self, config: TabPFNConfig):
        n_params_path = files("scan_benchmark.tabpfn").joinpath("n_params.json")
        with open(n_params_path, encoding="utf-8") as f:
            data = json.load(f)

        embedding_size = self._to_exponential(
            "embedding_size", config.embedding_size
        )
        num_layers = self._to_exponential(
            "num_layers", config.num_layers
        )

        for entry in data:
            cfg = entry["model_config"]

            if (
                    cfg["embedding_size"] == embedding_size and
                    cfg["num_layers"] == num_layers
            ):
                return entry["n_parameters"]

        raise ValueError(
            f"No n_parameters found for embedding_size={config.embedding_size}, "
            f"num_layers={config.num_layers}"
        )

    def flops(self, config: TabPFNConfig) -> float:
        flops_path = files("scan_benchmark.tabpfn").joinpath("flops.json")
        with open(flops_path, encoding="utf-8") as f:
            data = json.load(f)

        embedding_size = self._to_exponential(
            "embedding_size", config.embedding_size
        )
        num_layers = self._to_exponential(
            "num_layers", config.num_layers
        )

        for entry in data:
            cfg = entry["config"]

            if (
                    cfg["embedding_size"] == embedding_size
                    and cfg["num_layers"] == num_layers
            ):
                return entry["flops_per_cell"] * config.total_cells

    def _to_exponential(self, name: str, value):
        if name in self.default_exponential:
            return 2 ** value
        return value


if __name__ == "__main__":
    tabpfn_bench = TabPFNBenchmark(predictor_type=PerformancePredictorType.ENSEMBLE_XGB,
                                   device="auto")

    config = TabPFNConfig(
        total_cells=20,
        effective_batch_size=4,
        lr=0.0001,
        max_features=5,
        embedding_size=4,
        num_layers=0,
        num_datapoints_max=7,
    )

    result = tabpfn_bench.query_many([config])
    pprint(result)
