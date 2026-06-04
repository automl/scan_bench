import json
from pathlib import Path
from typing import Sequence, Callable

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

    def validate(self, dataset: BaseSurrogateDataset, sizes):
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

    def validate(self, dataset: BaseSurrogateDataset, sizes, out_path: Path, final_model_out_path: Path):
        X_test, y_test = dataset.get_test_data()
        test_bins = dataset._get_test_bins()
        has_bins = test_bins is not None

        max_n_cfg = max(sizes)

        for i, label in enumerate(self.labels):
            safe_label = label.replace("/", "_")

            overall_file_path = out_path / f"{safe_label}.json"
            by_bin_file_path = out_path / f"{safe_label}_by_bin.json"

            if overall_file_path.exists() and by_bin_file_path.exists():
                print(f"Skipping {label}, already exists.")
                continue

            model = self.models[label]
            print(f"Processing label '{label}' ({i + 1}/{len(self.labels)})")

            overall_results = {}

            for n_cfg in sizes:
                print(f"[{label}] Training with n_cfg={n_cfg}")

                X_sub, y_sub = dataset.get_train_subset(n_cfg)

                model.fit(X_sub, y_sub[:, i])

                full_pred = model.predict_with_uncertainty(X_test)
                y_pred = full_pred["mean"]

                overall_metrics = compute_regression_metrics(y_test[:, i], y_pred)
                overall_results[int(n_cfg)] = overall_metrics

                if has_bins and n_cfg == max_n_cfg:
                    by_bin_results = {
                        "n_cfg": int(n_cfg),
                        "metrics_by_bin": self.compute_metrics_by_group(
                            y_true=y_test[:, i],
                            y_pred=y_pred,
                            groups=test_bins,
                        ),
                    }
                    with open(by_bin_file_path, "w") as f:
                        json.dump(by_bin_results, f, indent=4)

                full_results_path = out_path / "full_pred.json"

                full_pred_json = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in full_pred.items()
                }

                full_pred_json["y_true"] = y_test[:, i].tolist()

                with open(full_results_path, "w") as f:
                    json.dump(full_pred_json, f, indent=4)

            with open(overall_file_path, "w") as f:
                json.dump(overall_results, f, indent=4)

            if final_model_out_path.exists() and any(final_model_out_path.iterdir()):
                print(f"Skipping final model save, it already exists: {final_model_out_path}")
                return

            print(f"{label} - Final training on all available data")
            X_all, y_all = dataset.get_all_data()
            print("Total checkpoints used:", len(X_all))
            model.fit(X_all, y_all[:, i])
            model.save(final_model_out_path / f"{safe_label}_models.joblib")

    def compute_metrics_by_group(self, y_true, y_pred, groups):
        results = {}
        groups = np.asarray(groups)

        for group in sorted(np.unique(groups)):
            mask = groups == group
            n = int(mask.sum())

            if n == 0:
                continue

            metrics = compute_regression_metrics(y_true[mask], y_pred[mask])
            metrics["n_samples"] = n
            results[str(group)] = metrics

        return results

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
