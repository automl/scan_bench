import os
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from scan_benchmark.vlm.divergence_surrogate.metrics import classification_metrics


def binary_xgb_ensemble(seed):
    models = []

    for depth in [3, 5, 9]:
        for n_estimators in [300, 500, 800]:
            for lr in [0.01, 0.05, 0.1]:
                models.append(
                    XGBClassifier(
                        max_depth=depth,
                        n_estimators=n_estimators,
                        learning_rate=lr,
                        subsample=0.9,
                        random_state=seed)
                )

    return models


class BinaryBaggingEnsemble:
    def __init__(self, seed: int = 42, model_dir: str = "scan_benchmark/vlm/divergence_surrogate/xgb_models",
                 load: bool = True):
        self.seed = seed
        self.model_dir = Path(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)

        self.models = []

        if load:
            self.load_models()
        else:
            self.models = binary_xgb_ensemble(self.seed)

    def __fit(self, X: np.ndarray, y: np.ndarray, save_models: bool = False):
        X = np.asarray(X)
        y = np.asarray(y)

        num_negative = (y == 0).sum()
        num_positive = (y == 1).sum()
        scale_pos_weight = num_negative / num_positive

        for i, model in enumerate(self.models):
            model.set_params(scale_pos_weight=scale_pos_weight)
            model.fit(X, y)

            if save_models:
                model_filename = os.path.join(self.model_dir, f"xgb_model_{i}.json")
                model.save_model(model_filename)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        preds = [m.predict(X) for m in self.models]
        preds = np.asarray(preds).T

        return np.mean(preds, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        probs = [m.predict_proba(X)[:, 1] for m in self.models]
        probs = np.asarray(probs).T

        return np.mean(probs, axis=1)

    def validate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5):
        y_prob = self.predict_proba(X)
        y_pred_class = (y_prob >= threshold).astype(int)
        classification_metrics(y, y_pred_class, y_prob, output_dir="scan_benchmark/vlm/divergence_surrogate/results")

    def load_models(self):
        model_files = sorted(self.model_dir.glob("xgb_model_*.json"))

        if not model_files:
            raise FileNotFoundError(f"No saved models found in {self.model_dir}")

        self.models = []
        for model_file in model_files:
            model = XGBClassifier()
            model.load_model(model_file)
            self.models.append(model)
