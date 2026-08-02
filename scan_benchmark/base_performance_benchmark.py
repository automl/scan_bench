from enum import Enum
from pathlib import Path

import numpy as np

from scan_benchmark.commons.predictors.autogluon import AutoGluonModel
from scan_benchmark.commons.predictors.ensembles import BaggingEnsemble, EnsembleType
from scan_benchmark.commons.predictors_core.pfn import TabPFNModel
from scan_benchmark.config_feature_mapper import ConfigFeatureMapper


class PerformancePredictorType(Enum):
    TABPFN = "tabpfn"
    TABPFN_HALF_A = "tabpfn_half_a"
    TABPFN_HALF_B = "tabpfn_half_b"
    ENSEMBLE_XGB = "ensemble_xgb"
    ENSEMBLE_LIGHTGBM = "ensemble_lightgbm"
    ENSEMBLE_MIX = "ensemble_mix"
    AUTOGLUON = "autogluon"


# TabPFN variants conditioned on one disjoint half of the collected data each. Running the
# same study with both gives an upper bound on the variance of the full-data surrogate.
TABPFN_DATA_HALVES = {
    PerformancePredictorType.TABPFN_HALF_A: 0,
    PerformancePredictorType.TABPFN_HALF_B: 1,
}

TABPFN_PREDICTOR_TYPES = {PerformancePredictorType.TABPFN, *TABPFN_DATA_HALVES}


def is_tabpfn(predictor_type: PerformancePredictorType) -> bool:
    return predictor_type in TABPFN_PREDICTOR_TYPES


def get_data_half(predictor_type: PerformancePredictorType) -> int | None:
    return TABPFN_DATA_HALVES.get(predictor_type)


class BasePerformanceBenchmark:
    TARGET_ENUM = None

    def __init__(self, surrogate_dataset, target, model_path: Path = None,
                 predictor_type: PerformancePredictorType = PerformancePredictorType.TABPFN, device="auto",
                 additional_runs_path: Path = None, data_split_seed: int = 42):
        self.surrogate_dataset = surrogate_dataset
        self.target = self._normalize_target(target)
        self.target_idx = self.surrogate_dataset.targets.index(self.target)

        self.predictor_type = predictor_type
        self.model_path = model_path
        self.additional_runs_path = additional_runs_path
        self.data_half = get_data_half(predictor_type)
        self.data_split_seed = data_split_seed

        self.config_feature_mapper = ConfigFeatureMapper(
            feature_order=self.surrogate_dataset.features,
            apply_log=self.surrogate_dataset.apply_log_transform,
            log_columns=self.surrogate_dataset.DEFAULT_LOG_COLUMNS,
            exponential_columns=self.surrogate_dataset.DEFAULT_EXPONENTIAL
        )

        self.performance_surrogate = self._build_performance_surrogate(predictor_type, device)
        self._prepare_surrogate()

    def _normalize_target(self, target):
        if self.TARGET_ENUM is None:
            raise ValueError("TARGET_ENUM must be set in subclass.")

        if target is None:
            return self.TARGET_ENUM.default()

        return target

    def _get_model_path(self, target: str) -> Path:
        if self.model_path is None:
            raise ValueError("model_path must be provided for saved surrogate predictors.")

        safe_target = target.replace("/", "_")
        model_path = self.model_path / f"{safe_target}_models.joblib"

        return model_path

    def _prepare_surrogate(self):
        if is_tabpfn(self.predictor_type):
            X_train, y_train = self.surrogate_dataset.get_all_data(
                additional_csv_path=self.additional_runs_path,
                half=self.data_half,
                n_halves=len(TABPFN_DATA_HALVES),
                split_seed=self.data_split_seed,
            )
            self.performance_surrogate.fit(X_train, y_train)

        elif self.predictor_type == PerformancePredictorType.AUTOGLUON:
            surrogate_model_path = self.model_path / self.target / "final_run"

            if not surrogate_model_path.exists():
                X_train, y_train = self.surrogate_dataset.get_all_data(additional_csv_path=self.additional_runs_path)
                self.performance_surrogate.label = self.target
                self.performance_surrogate.time_limit = 4 * 30 * 60
                self.performance_surrogate.fit_and_save(X_train, y_train, surrogate_model_path)

            self.performance_surrogate.load(surrogate_model_path)

        else:
            surrogate_model_path = self._get_model_path(self.target)
            if not surrogate_model_path.exists():
                X_train, y_train = self.surrogate_dataset.get_all_data(additional_csv_path=self.additional_runs_path)
                self.performance_surrogate.fit_and_save(X_train, y_train, surrogate_model_path)

            self.performance_surrogate.load(surrogate_model_path)

        return self.performance_surrogate

    def _format_prediction(self, mean, uncertainty):
        return {
            "mean": round(float(mean), 3),
            "uncertainty": round(float(uncertainty), 3),
        }

    def _predict_performance(self, config):
        row = self.config_feature_mapper.to_features(config)
        pred = self.performance_surrogate.predict_with_uncertainty(row)

        return self._format_prediction(
            pred["mean"][0],
            pred["uncertainty"][0],
        )

    def _configs_to_surrogate_matrix(self, configs):
        rows = [self.config_feature_mapper.to_features(cfg) for cfg in configs]
        if len(rows) == 0:
            raise ValueError("No valid configurations provided.")
        return np.vstack(rows)

    def _predict_performance_many(self, configs):
        X_query = self._configs_to_surrogate_matrix(configs)
        pred = self.performance_surrogate.predict_with_uncertainty(X_query)

        return [
            self._format_prediction(mean, uncertainty)
            for mean, uncertainty in zip(
                pred["mean"],
                pred["uncertainty"],
            )
        ]

    def _build_performance_surrogate(
            self,
            predictor_type: PerformancePredictorType,
            device="auto",
    ):
        if is_tabpfn(predictor_type):
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

        if predictor_type == PerformancePredictorType.AUTOGLUON:
            return AutoGluonModel()
