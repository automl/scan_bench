import json
from pathlib import Path
from typing import Sequence, Callable

import numpy as np

from surrogate_benchmark.vlm.performance_surrogate.data import SurrogateDataset
from surrogate_benchmark.metrics.metrics import compute_regression_metrics


class SurrogateModel:
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def validate(self, dataset: SurrogateDataset, sizes):
        X_test, y_test = dataset.get_test_data()
        results = {}

        for n_cfg in sizes:
            X_sub, y_sub = dataset.get_train_subset(n_cfg)
            self.fit(X_sub, y_sub)

            y_pred = self.predict(X_test)

            results[int(n_cfg)] = compute_regression_metrics(y_test, y_pred)

        return results


class MultiLabelSurrogateModel:
    def __init__(
            self,
            labels: Sequence[str],
            model_factory: Callable[[str], SurrogateModel],
    ):
        self.labels = list(labels)
        self.models = {
            label: model_factory(label) for label in self.labels
        }
        self.is_fitted = False

    def validate(self, dataset: SurrogateDataset, sizes, out_path: Path):
        X_test, y_test = dataset.get_test_data()

        for i, label in enumerate(self.labels):
            file_path = out_path / f"{label}.json"

            if file_path.exists():
                print(f"Skipping {label}, already exists.")
                continue

            model = self.models[label]
            label_results = {}

            for n_cfg in sizes:
                X_sub, y_sub = dataset.get_train_subset(n_cfg)

                model.fit(X_sub, y_sub[:, i])

                y_pred = model.predict(X_test)
                metrics = compute_regression_metrics(y_test[:, i], y_pred)

                label_results[int(n_cfg)] = metrics

            with open(file_path, "w") as f:
                json.dump(label_results, f, indent=4)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        if y.ndim != 2 or y.shape[1] != len(self.labels):
            raise ValueError(
                f"Expected y shape (N, {len(self.labels)}), got {y.shape}."
            )

        for i, label in enumerate(self.labels):
            self.models[label].fit(X, y[:, i])
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Surrogate must be fitted before predict().")
        X = np.asarray(X)
        preds = [np.asarray(self.models[label].predict(X)).reshape(-1) for label in self.labels]
        return np.stack(preds, axis=1)

    def get_predictor(self, label: str) -> SurrogateModel:
        return self.models[label]
