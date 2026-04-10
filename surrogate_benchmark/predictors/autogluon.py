import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from surrogate_benchmark.core.base import SurrogateModel


class AutoGluonModel(SurrogateModel):

    def __init__(
            self,
            features: list[str],
            label: str = "val_loss",
            time_limit: int = 30 * 60,
            presets: str = "best_quality",
            base_path: str = "AutogluonModels",
            use_manual_bagging: bool = True,
            num_stack_levels: int = 1,
            num_bag_folds: int = 8,
            num_bag_sets: int = 2,
    ):
        self.features = features
        self.label = label
        self.time_limit = time_limit
        self.presets = presets
        self.base_path = base_path

        self.use_manual_bagging = use_manual_bagging
        self.num_stack_levels = num_stack_levels
        self.num_bag_folds = num_bag_folds
        self.num_bag_sets = num_bag_sets

        self.predictor = None
        self.run_id = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.run_id += 1

        X = np.asarray(X)
        y = np.asarray(y)

        train_df = pd.DataFrame(X, columns=self.features)
        train_df[self.label] = y

        path = f"{self.base_path}/run_{self.run_id}"

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
        X_df = pd.DataFrame(X, columns=self.features)

        return self.predictor.predict(X_df).to_numpy()
