import numpy as np
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from scan_benchmark.commons.predictors_core.surrogate import SurrogateModel


class TabPFNModel(SurrogateModel):
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = TabPFNRegressor.create_default_for_version(
                        ModelVersion.V2_5,
                        device=device,
                        random_state=42,
                        ignore_pretraining_limits=True,
                        fit_mode="fit_with_cache",
                    )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def save(self, path):
        print("TabPFN performs in-context learning, no need to save models!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _align_quantiles_by_sample(self, quantiles: np.ndarray, n_samples: int) -> np.ndarray:
        quantiles = np.asarray(quantiles)

        if quantiles.ndim == 1:
            if n_samples != 1:
                raise ValueError(
                    f"Expected quantiles for {n_samples} samples, got 1D shape {quantiles.shape}."
                )
            return quantiles.reshape(1, -1)

        if quantiles.ndim != 2:
            raise ValueError(f"Unexpected quantiles shape: {quantiles.shape}")

        if quantiles.shape[0] == n_samples:
            return quantiles

        if quantiles.shape[1] == n_samples:
            return quantiles.T

        raise ValueError(
            f"Could not align quantiles shape {quantiles.shape} with n_samples={n_samples}."
        )

    def predict_with_uncertainty(self, X: np.ndarray) -> dict[str, np.ndarray]:
        outputs = self.model.predict(X, output_type="main")

        if not isinstance(outputs, dict):
            mean = np.asarray(outputs).reshape(-1)
            return {"mean": mean}

        mean = np.asarray(outputs.get("mean")).reshape(-1)
        n_samples = mean.shape[0]

        result: dict[str, np.ndarray] = {
            "mean": mean,
        }

        if "quantiles" in outputs:
            quantiles = self._align_quantiles_by_sample(outputs["quantiles"], n_samples)
            result["quantiles"] = quantiles
            result["uncertainty"] = quantiles[:, -1] - quantiles[:, 0]

        return result
