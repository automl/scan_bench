from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from scan_benchmark.commons.predictors_core.surrogate import SurrogateModel


class AutoGluonModel(SurrogateModel):

    def __init__(
            self,
            label: str = "val_loss",
            time_limit: int = 30 * 60,
            presets: str = "best_quality",
            base_path: str = "AutogluonModels",
            use_manual_bagging: bool = True,
            num_stack_levels: int = 1,
            num_bag_folds: int = 8,
            num_bag_sets: int = 2,
            fold: int = 1,
    ):
        self.label = label
        self.time_limit = time_limit
        self.presets = presets
        self.base_path = base_path

        self.use_manual_bagging = use_manual_bagging
        self.num_stack_levels = num_stack_levels
        self.num_bag_folds = num_bag_folds
        self.num_bag_sets = num_bag_sets

        self.predictor = None
        self.fold = fold

    def fit(self, X: np.ndarray, y: np.ndarray, path: Path = None, final: bool = False):
        X = np.asarray(X)
        y = np.asarray(y)

        train_df = pd.DataFrame(X)
        train_df[self.label] = y
        if path is None:
            path = f"{self.base_path}/{self.label}/fold_{self.fold}" if not final else f"{self.base_path}/{self.label}/final_run"

        self.predictor = TabularPredictor(
            label=self.label,
            problem_type="regression",
            eval_metric="rmse",
            path=path,
        )

        fit_kwargs = {
            "train_data": train_df,
            "time_limit": self.time_limit,
            "presets": self.presets,
            "fit_strategy": "sequential",
            "excluded_model_types": ["FASTAI"],
        }

        if self.use_manual_bagging:
            fit_kwargs.update({
                "dynamic_stacking": False,
                "num_stack_levels": self.num_stack_levels,
                "num_bag_folds": self.num_bag_folds,
                "num_bag_sets": self.num_bag_sets,
            })

        self.predictor.fit(**fit_kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        X_df = pd.DataFrame(X)

        return self.predictor.predict(X_df).to_numpy()

    def predict_with_uncertainty(self, X: np.ndarray):
        X = np.asarray(X)
        X_df = pd.DataFrame(X)

        preds = []

        for model_name in self.predictor.model_names():
            if model_name.startswith("WeightedEnsemble"):
                continue

            preds.append(
                self.predictor.predict(X_df, model=model_name).to_numpy()
            )

        preds = np.array(preds)

        mean = np.mean(preds, axis=0)
        uncertainty = np.std(preds, axis=0)

        return {
            "mean": mean,
            "uncertainty": uncertainty,
        }

    def load(self, path: str):
        self.predictor = TabularPredictor.load(path)
        self.predictor.persist(
            models="best",
            with_ancestors=True,
            max_memory=None,
        )

    def fit_and_save(self, X: np.ndarray, y: np.ndarray, path: Path):
        self.fit(X, y, path)
