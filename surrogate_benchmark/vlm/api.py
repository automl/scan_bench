import json
from importlib.resources import files
from pprint import pprint

import numpy as np
from tqdm import tqdm

from surrogate_benchmark.config_feature_mapper import ConfigFeatureMapper
from surrogate_benchmark.predictors_core.pfn import TabPFNModel
from surrogate_benchmark.vlm.config import VLMConfig, Target
from surrogate_benchmark.vlm.divergence_surrogate.data import DivergenceDataset
from surrogate_benchmark.vlm.divergence_surrogate.predictors.xgb import BinaryBaggingEnsemble
from surrogate_benchmark.vlm.performance_surrogate.data import SurrogateDataset


class VLMBenchmark:
    def __init__(
            self,
            targets: list[Target] | None = None,
            device: str = "auto",
    ):
        targets = Target.all() if targets is None else [t.value for t in targets]

        train_path = files("surrogate_benchmark.vlm.performance_surrogate") \
            .joinpath("splits/train.csv")

        test_path = files("surrogate_benchmark.vlm.performance_surrogate") \
            .joinpath("splits/test.csv")

        self.surrogate_dataset = SurrogateDataset(
            train_csv_path=str(train_path),
            test_csv_path=str(test_path),
            targets=targets,
            seed=42,
            include_intermediate_points=True,
        )

        self.performance_config_feature_mapper = ConfigFeatureMapper(
            feature_order=self.surrogate_dataset.features,
            apply_log=self.surrogate_dataset.apply_log_transform,
            log_columns=self.surrogate_dataset.DEFAULT_LOG_COLUMNS,
        )

        self.divergence_config_feature_mapper = ConfigFeatureMapper(
            feature_order=DivergenceDataset.DEFAULT_FEATURES,
            apply_log=self.surrogate_dataset.apply_log_transform,
            log_columns=self.surrogate_dataset.DEFAULT_LOG_COLUMNS,
        )

        self.targets = self.surrogate_dataset.targets

        model_dir = files("surrogate_benchmark.vlm.divergence_surrogate") \
            .joinpath("xgb_models")

        self.divergence_surrogate = BinaryBaggingEnsemble(model_dir=str(model_dir))

        self.performance_surrogate = TabPFNModel(device=device)

    def query(self, config: VLMConfig) -> dict:
        divergence_prob, failed = self._predict_divergence(config)
        stats = self.get_model_stats(config)

        return {
            "failed": failed,
            "divergence_probability": round(divergence_prob, 3),
            "predictions": None if failed else self._predict_performance(config),
            "model_stats": stats,
        }

    def get_model_stats(self, config: VLMConfig) -> dict:
        stats_key = f"text_{config.text_width}_vision_{config.vision_width}"

        path = files("surrogate_benchmark.vlm").joinpath("gflops_params.json")

        with open(str(path), "r", encoding="utf-8") as f:
            models_stats = json.load(f)

        model_stats = models_stats.get(stats_key)
        if model_stats is None:
            return {}

        gflops_per_sample = model_stats.get("gflops_per_sample")

        return {
            "gflops_per_sample": gflops_per_sample,
            "total_gflops": gflops_per_sample * config.total_samples_planned * config.training_progress,
            "params": model_stats.get("params"),
            "vision_params": model_stats.get("vision_params"),
            "text_params": model_stats.get("text_params"),
            "text_params_non_emb": model_stats.get("text_params_non_emb"),
        }

    def _predict_divergence(self, config: VLMConfig) -> tuple[float, bool]:
        row = self.divergence_config_feature_mapper.to_features(config)
        prob = float(self.divergence_surrogate.predict_proba(row)[0])
        return prob, prob >= 0.5

    def _predict_performance(self, config: VLMConfig) -> dict:
        row = self.performance_config_feature_mapper.to_features(config)
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

    def _configs_to_divergence_matrix(self, configs: list[VLMConfig]) -> np.ndarray:
        rows = [self.divergence_config_feature_mapper.to_features(cfg) for cfg in configs]
        return np.vstack(rows)

    def _configs_to_surrogate_matrix(self, configs: list[VLMConfig]) -> np.ndarray:
        rows = [self.performance_config_feature_mapper.to_features(cfg) for cfg in configs]
        return np.vstack(rows)

    def _predict_divergence_many(self, configs: list[VLMConfig]) -> tuple[np.ndarray, np.ndarray]:
        X = self._configs_to_divergence_matrix(configs)
        probs = np.asarray(self.divergence_surrogate.predict_proba(X)).reshape(-1)
        failed = probs >= 0.5
        return probs, failed

    def _predict_performance_many(self, configs: list[VLMConfig]) -> list[dict]:
        X_query = self._configs_to_surrogate_matrix(configs)
        X_train, y_train = self.surrogate_dataset.get_all_data()

        predictions_per_config = [{} for _ in configs]

        for i, target in enumerate(tqdm(self.targets, desc="Evaluating targets")):
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

    def query_many(self, configs: list[VLMConfig]) -> list[dict]:
        divergence_probs, failed_mask = self._predict_divergence_many(configs)
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
    vlm_bench = VLMBenchmark(targets=[Target.VAL_LOSS])

    config = VLMConfig(
        lr=1e-4,
        wd=0.01,
        beta1=0.9,
        beta2=0.98,
        warmup_fraction=0.05,
        eps=1e-8,
        vision_width=256,
        text_width=256,
        total_samples_planned=55000000,
        training_progress=1.0
    )

    config2 = VLMConfig(
        lr=0.03,
        wd=0.01,
        beta1=0.9,
        beta2=0.98,
        warmup_fraction=0.05,
        eps=1e-8,
        vision_width=384,
        text_width=256,
        total_samples_planned=12800000,
        training_progress=0.5
    )

    results = vlm_bench.query_many([config, config2])

    pprint(results)
