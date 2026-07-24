import json
from importlib.resources import files
from pprint import pprint

import numpy as np

from scan_benchmark.base_performance_benchmark import BasePerformanceBenchmark, PerformancePredictorType
from scan_benchmark.config_feature_mapper import ConfigFeatureMapper
from scan_benchmark.vlm.config import VLMConfig, VLMTarget
from scan_benchmark.vlm.divergence_surrogate.data import DivergenceDataset
from scan_benchmark.vlm.divergence_surrogate.predictors.xgb import BinaryBaggingEnsemble
from scan_benchmark.vlm.performance_surrogate.data import VLMSurrogateDataset


class VLMBenchmark(BasePerformanceBenchmark):
    TARGET_ENUM = VLMTarget

    def __init__(self, target=None, predictor_type: PerformancePredictorType = PerformancePredictorType.TABPFN,
                 autogluon_model_path: str | None = None, use_divergence_predictor: bool = True, device="auto"):
        train_path = files("scan_benchmark.vlm.performance_surrogate").joinpath("splits/train_fold_1.csv")
        test_path = files("scan_benchmark.vlm.performance_surrogate").joinpath("splits/test_fold_1.csv")
        additional_runs_path = files("scan_benchmark.vlm").joinpath("large_runs.csv")

        dataset = VLMSurrogateDataset(
            train_csv_path=str(train_path),
            test_csv_path=str(test_path),
            targets=target,
            seed=42,
            include_intermediate_points=True,
            eval_on_intermediate_points=True
        )

        if predictor_type == PerformancePredictorType.AUTOGLUON:
            saved_model_path = (
                autogluon_model_path
                if autogluon_model_path is not None
                else files("scan_benchmark.vlm.performance_surrogate").joinpath(
                    "saved_models",
                    "predictors",
                    predictor_type.value,
                    "seed_42",
                    "fit_with_intermediate",
                    "pred_with_intermediate",
                    "auto",
                )
            )

        else:
            saved_model_path = files("scan_benchmark.vlm.performance_surrogate").joinpath(
                "saved_models",
                "predictors",
                predictor_type.value,
                f"seed_42",
                "fit_with_intermediate",
                "pred_with_intermediate",
            )

        super().__init__(
            target=target,
            surrogate_dataset=dataset,
            model_path=saved_model_path,
            predictor_type=predictor_type,
            device=device,
            additional_runs_path=additional_runs_path
        )

        self.use_divergence_predictor = use_divergence_predictor

        if self.use_divergence_predictor:
            self.divergence_config_feature_mapper = ConfigFeatureMapper(
                feature_order=DivergenceDataset.DEFAULT_FEATURES,
                apply_log=self.surrogate_dataset.apply_log_transform,
                log_columns=self.surrogate_dataset.DEFAULT_LOG_COLUMNS,
            )

            model_dir = files(
                "scan_benchmark.vlm.divergence_surrogate"
            ).joinpath("xgb_models")

            self.divergence_surrogate = BinaryBaggingEnsemble(
                model_dir=str(model_dir)
            )

    def query(self, config: VLMConfig) -> dict:
        if self.use_divergence_predictor:
            divergence_prob, failed = self._predict_divergence(config)
        else:
            divergence_prob, failed = 0.0, False
        stats = self.get_model_stats(config)

        return {
            "failed": bool(failed),
            "divergence_probability": round(float(divergence_prob), 3),
            "predictions": None if failed else self._predict_performance(config),
            "model_stats": stats,
        }

    def get_model_stats(self, config: VLMConfig) -> dict:
        stats_key = f"text_{config.text_width}_vision_{config.vision_width}"
        path = files("scan_benchmark.vlm").joinpath("flops_params.json")

        with open(path, encoding="utf-8") as f:
            model_stats = json.load(f).get(stats_key)

        if model_stats is None:
            return {}

        flops_per_sample = model_stats.get("flops_per_sample")

        return {
            "flops_per_sample": flops_per_sample,
            "total_flops": flops_per_sample * config.total_samples_planned * config.training_progress,
            "params": model_stats.get("params"),
            "vision_params": model_stats.get("vision_params"),
            "text_params": model_stats.get("text_params"),
            "text_params_non_emb": model_stats.get("text_params_non_emb"),
        }

    def _predict_divergence(self, config: VLMConfig) -> tuple[float, bool]:
        row = self.divergence_config_feature_mapper.to_features(config)
        prob = float(self.divergence_surrogate.predict_proba(row)[0])
        return prob, prob >= 0.5

    def _configs_to_divergence_matrix(self, configs: list[VLMConfig]) -> np.ndarray:
        rows = [self.divergence_config_feature_mapper.to_features(cfg) for cfg in configs]
        return np.vstack(rows)

    def _predict_divergence_many(self, configs: list[VLMConfig]) -> tuple[np.ndarray, np.ndarray]:
        X = self._configs_to_divergence_matrix(configs)
        probs = np.asarray(self.divergence_surrogate.predict_proba(X)).reshape(-1)
        failed = probs >= 0.5
        return probs, failed

    def query_many(self, configs: list[VLMConfig]) -> list[dict]:
        if self.use_divergence_predictor:
            divergence_probs, failed_mask = self._predict_divergence_many(configs)
        else:
            divergence_probs = np.full(len(configs), 0.0)
            failed_mask = np.zeros(len(configs), dtype=bool)
        stats_list = [self.get_model_stats(cfg) for cfg in configs]

        valid_idx = [i for i, f in enumerate(failed_mask) if not f]
        valid_configs = [configs[i] for i in valid_idx]

        valid_preds = self._predict_performance_many(valid_configs)
        pred_map = {i: p for i, p in zip(valid_idx, valid_preds)}

        return [
            {
                "failed": bool(failed),
                "divergence_probability": round(float(prob), 3),
                "predictions": None if failed else pred_map[i],
                "model_stats": stats,
            }
            for i, (prob, failed, stats) in enumerate(zip(divergence_probs, failed_mask, stats_list))
        ]


# simple example on how to use the surrogate
if __name__ == "__main__":
    vlm_bench = VLMBenchmark(predictor_type=PerformancePredictorType.TABPFN, use_divergence_predictor=False,
                             device="cpu")

    config = VLMConfig(
        lr=1e-4,
        wd=0.01,
        beta1=0.9,
        beta2=0.98,
        warmup_fraction=0.05,
        eps=1e-8,
        vision_width=32,
        text_width=32,
        total_samples_planned=600_000,
        training_progress=1.0
    )

    config2 = VLMConfig(
        lr=0.0003,
        wd=0.01,
        beta1=0.9,
        beta2=0.98,
        warmup_fraction=0.05,
        eps=1e-8,
        vision_width=1024,
        text_width=1024,
        total_samples_planned=200_000_000,
        training_progress=1.0
    )

    results = vlm_bench.query_many([config, config2])

    pprint(results)
