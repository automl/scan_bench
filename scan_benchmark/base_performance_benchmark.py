from enum import Enum
from pathlib import Path

import numpy as np

from scan_benchmark.commons.predictors.ensembles import BaggingEnsemble, EnsembleType
from scan_benchmark.commons.predictors_core.pfn import TabPFNModel
from scan_benchmark.config_feature_mapper import ConfigFeatureMapper


class PerformancePredictorType(Enum):
    TABPFN = "tabpfn"
    ENSEMBLE_XGB = "ensemble_xgb"
    ENSEMBLE_LIGHTGBM = "ensemble_lightgbm"
    ENSEMBLE_MIX = "ensemble_mix"


class BasePerformanceBenchmark:
    TARGET_ENUM = None

    def __init__(self, surrogate_dataset, model_path: Path = None,
                 predictor_type: PerformancePredictorType = PerformancePredictorType.TABPFN, device="auto"):
        self.surrogate_dataset = surrogate_dataset
        self.targets = surrogate_dataset.targets
        self.predictor_type = predictor_type
        self.model_path = model_path
        self.performance_surrogate = self._build_performance_surrogate(predictor_type, device)

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

    def _get_model_path(self, target: str) -> Path:
        if self.model_path is None:
            raise ValueError("model_path must be provided for saved surrogate predictors.")

        safe_target = target.replace("/", "_")
        return self.model_path / f"{safe_target}_models.joblib"

    def _prepare_surrogate_for_target(self, target: str, target_idx: int):
        if self.predictor_type == PerformancePredictorType.TABPFN:
            X_train, y_train = self.surrogate_dataset.get_all_data()
            self.performance_surrogate.fit(X_train, y_train[:, target_idx])
        else:
            self.performance_surrogate.load(self._get_model_path(target))

        return self.performance_surrogate

    def _format_prediction(self, mean, uncertainty):
        return {
            "mean": round(float(mean), 3),
            "uncertainty": round(float(uncertainty), 3),
        }

    def _predict_performance(self, config):
        row = self.config_feature_mapper.to_features(config)
        predictions = {}

        for i, target in enumerate(self.targets):
            surrogate = self._prepare_surrogate_for_target(target, i)
            pred = surrogate.predict_with_uncertainty(row)

            predictions[target] = self._format_prediction(
                pred["mean"][0],
                pred["uncertainty"][0],
            )

        return predictions

    def _configs_to_surrogate_matrix(self, configs):
        rows = [self.config_feature_mapper.to_features(cfg) for cfg in configs]
        if len(rows) == 0:
            raise ValueError("No valid configurations provided.")
        return np.vstack(rows)

    def _predict_performance_many(self, configs):
        X_query = self._configs_to_surrogate_matrix(configs)
        predictions_per_config = [{} for _ in configs]

        for i, target in enumerate(self.targets):
            surrogate = self._prepare_surrogate_for_target(target, i)
            pred = surrogate.predict_with_uncertainty(X_query)

            means = pred["mean"]
            uncertainties = pred["uncertainty"]

            for j in range(len(configs)):
                predictions_per_config[j][target] = self._format_prediction(
                    means[j],
                    uncertainties[j],
                )

        return predictions_per_config

    def _build_performance_surrogate(
            self,
            predictor_type: PerformancePredictorType,
            device="auto",
    ):
        if predictor_type == PerformancePredictorType.TABPFN:
            return TabPFNModel(device=device)

        if predictor_type == PerformancePredictorType.ENSEMBLE_XGB:
            return BaggingEnsemble(
                ensemble_type=EnsembleType.XGB
            )

        if predictor_type == PerformancePredictorType.ENSEMBLE_LIGHTGBM:
            return BaggingEnsemble(
                ensemble_type=EnsembleType.LIGHTGBM
            )

        if predictor_type == PerformancePredictorType.ENSEMBLE_MIX:
            return BaggingEnsemble(
                ensemble_type=EnsembleType.MIX
            )

        raise ValueError(f"Unknown predictor type: {predictor_type}")
