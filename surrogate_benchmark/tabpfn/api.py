from importlib.resources import files

from tqdm import tqdm

from surrogate_benchmark.config_feature_mapper import ConfigFeatureMapper
from surrogate_benchmark.predictors_core.pfn import TabPFNModel
from surrogate_benchmark.tabpfn.config import TabPFNConfig
from surrogate_benchmark.tabpfn.performance_surrogate.data import SurrogateDataset


class TabPFNBenchmark:
    def __init__(
            self,
            device: str = "auto",
    ):
        train_path = files("surrogate_benchmark.tabpfn.performance_surrogate") \
            .joinpath("splits/train.csv")

        test_path = files("surrogate_benchmark.tabpfn.performance_surrogate") \
            .joinpath("splits/test.csv")

        self.surrogate_dataset = SurrogateDataset(
            train_csv_path=str(train_path),
            test_csv_path=str(test_path),
            seed=42,
        )

        self.targets = self.surrogate_dataset.targets

        self.performance_surrogate = TabPFNModel(device=device)

        self.config_feature_mapper = ConfigFeatureMapper(
            feature_order=self.surrogate_dataset.features,
            apply_log=self.surrogate_dataset.apply_log_transform,
            log_columns=self.surrogate_dataset.DEFAULT_LOG_COLUMNS,
        )

    def _predict_performance(self, config: TabPFNConfig) -> dict:
        row = self.config_feature_mapper.to_features(config)
        X, y = self.surrogate_dataset.get_all_data()

        predictions = {}

        for i, target in enumerate(tqdm(self.targets, desc="Evaluating targets")):
            y_target = y[:, i]

            self.performance_surrogate.fit(X, y_target)
            pred = self.performance_surrogate.predict_with_uncertainty(row)

            predictions[target] = {
                "mean": round(float(pred["mean"][0]), 3),
                "uncertainty": round(float(pred["uncertainty_width"][0]), 3)
            }

        return predictions

    def query(self, config: TabPFNConfig) -> dict:
        predictions = self._predict_performance(config)
        return {"predictions": predictions}


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

    result = tabpfn_bench.query(config)
    print(result)
