from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from scan_benchmark.commons.predictors_core.surrogate import SurrogateModel


def xgb_ensemble(seed):
    ensemble_list = []
    for depth in [5, 9, 3]:
        for n_estimators in [200, 500, 800]:
            for learning_rate in [0.01, 0.1, 1.0]:
                ensemble_list.append(
                    xgb.XGBRegressor(
                        max_depth=depth,
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        eval_metric="rmse",
                        random_state=seed,
                    )
                )
    return ensemble_list


def lightgbm_ensemble(seed):
    ensemble_list = []
    for depth in [5, 9, 3]:
        for n_estimators in [200, 500, 800]:
            for learning_rate in [0.01, 0.1, 0.001, 1.0]:
                ensemble_list.append(
                    LGBMRegressor(
                        max_depth=depth,
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        random_state=seed,
                    )
                )
    return ensemble_list


def mix_ensemble(seed):
    return [
        xgb.XGBRegressor(),
        xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.01, random_state=seed),
        xgb.XGBRegressor(n_estimators=800, max_depth=3, learning_rate=0.1, random_state=seed),
        xgb.XGBRegressor(n_estimators=200, max_depth=9, learning_rate=1, random_state=seed),
        LGBMRegressor(random_state=seed),
        LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.01, random_state=seed),
        LGBMRegressor(n_estimators=800, max_depth=3, learning_rate=0.1, random_state=seed),
        LGBMRegressor(n_estimators=200, max_depth=9, learning_rate=1, random_state=seed),
        LinearRegression(),
        Ridge(random_state=seed),
        RandomForestRegressor(n_estimators=400, max_depth=5, random_state=seed),
    ]


class EnsembleType(Enum):
    XGB = "xgb"
    LIGHTGBM = "lightgbm"
    MIX = "mix"


class BaggingEnsemble(SurrogateModel):
    def __init__(self, ensemble_type: EnsembleType, seed: int = 42):
        self.seed = seed
        self.ensemble_type = ensemble_type

        if ensemble_type == EnsembleType.XGB:
            self.models = xgb_ensemble(seed)
        elif ensemble_type == EnsembleType.LIGHTGBM:
            self.models = lightgbm_ensemble(seed)
        elif ensemble_type == EnsembleType.MIX:
            self.models = mix_ensemble(seed)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        for i, model in enumerate(self.models):
            print(f"Training ensemble member {i + 1}/{len(self.models)}")
            model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        preds = [m.predict(X) for m in self.models]
        preds = np.asarray(preds).T
        return np.mean(preds, axis=1)

    def predict_with_uncertainty(self, X: np.ndarray):
        preds = [m.predict(X) for m in self.models]
        preds = np.asarray(preds).T

        mean_pred = np.mean(preds, axis=1)
        uncertainty = np.std(preds, axis=1)

        return {
            "mean": mean_pred,
            "uncertainty": uncertainty
        }

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, path)

    def load(self, path: Path):
        self.models = joblib.load(path)

    def fit_and_save(self, X: np.ndarray, y: np.ndarray, path: Path):
        self.fit(X, y)
        self.save(path)
