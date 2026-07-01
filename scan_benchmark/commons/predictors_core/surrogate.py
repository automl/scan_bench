from pathlib import Path

import numpy as np

from scan_benchmark.commons.metrics.metrics import compute_regression_metrics
from scan_benchmark.dataset import BaseSurrogateDataset


class SurrogateModel:
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_with_uncertainty(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path):
        raise NotImplementedError

    def fit_and_save(self, X: np.ndarray, y: np.ndarray, path: Path):
        raise NotImplementedError

    def validate(self, dataset: BaseSurrogateDataset, sizes):
        X_test, y_test = dataset.get_test_data()
        results = {}

        for n_cfg in sizes:
            X_sub, y_sub = dataset.get_train_subset(n_cfg)
            self.fit(X_sub, y_sub)

            y_pred = self.predict(X_test)

            results[int(n_cfg)] = compute_regression_metrics(y_test, y_pred)

        return results
