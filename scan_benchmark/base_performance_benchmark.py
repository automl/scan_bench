import numpy as np

from scan_benchmark.commons.predictors_core.pfn import TabPFNModel
from scan_benchmark.config_feature_mapper import ConfigFeatureMapper


class BasePerformanceBenchmark:
    TARGET_ENUM = None

    def __init__(self, surrogate_dataset, device="auto"):
        self.surrogate_dataset = surrogate_dataset
        self.targets = surrogate_dataset.targets
        self.performance_surrogate = TabPFNModel(device=device)

        self.config_feature_mapper = ConfigFeatureMapper(
            feature_order=self.surrogate_dataset.features,
            apply_log=self.surrogate_dataset.apply_log_transform,
            log_columns=self.surrogate_dataset.DEFAULT_LOG_COLUMNS,
        )

    def _normalize_targets(self, targets):
        if self.TARGET_ENUM is None:
            raise ValueError("TARGET_ENUM must be set in subclass.")

        if targets is None:
            return self.TARGET_ENUM.all()

        return [t.value for t in targets]

    def _predict_performance(self, config):
        row = self.config_feature_mapper.to_features(config)
        X, y = self.surrogate_dataset.get_all_data()

        predictions = {}
        for i, target in enumerate(self.targets):
            y_target = y[:, i]
            self.performance_surrogate.fit(X, y_target)
            pred = self.performance_surrogate.predict_with_uncertainty(row)

            predictions[target] = {
                "mean": round(float(pred["mean"][0]), 3),
                "uncertainty": round(float(pred["uncertainty_width"][0]), 3),
            }

        return predictions

    def _configs_to_surrogate_matrix(self, configs):
        rows = [self.config_feature_mapper.to_features(cfg) for cfg in configs]
        if len(rows) == 0:
            raise ValueError("No valid configurations provided.")
        return np.vstack(rows)

    def _predict_performance_many(self, configs):
        X_query = self._configs_to_surrogate_matrix(configs)
        X_train, y_train = self.surrogate_dataset.get_all_data()

        predictions_per_config = [{} for _ in configs]

        for i, target in enumerate(self.targets):
            y_target = y_train[:, i]

            self.performance_surrogate.fit(X_train, y_target)
            pred = self.performance_surrogate.predict_with_uncertainty(X_query)

            means = pred["mean"]
            widths = pred.get("uncertainty_width")

            for j in range(len(configs)):
                predictions_per_config[j][target] = {
                    "mean": round(float(means[j]), 3),
                    "uncertainty": round(float(widths[j]), 3) if widths is not None else None,
                }

        return predictions_per_config
