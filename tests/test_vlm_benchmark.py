import numpy as np

from scan_benchmark.vlm.config import VLMConfig, VLMTarget
from scan_benchmark.vlm.performance_surrogate.data import VLMSurrogateDataset


class FakePerformanceModel:
    def __init__(self, device="auto"):
        self.device = device

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict_with_uncertainty(self, X):
        n = np.asarray(X).shape[0]
        return {
            "mean": np.arange(n, dtype=float) + 0.4321,
            "uncertainty": np.full(n, 0.1111),
        }


class FakeVLMDataset:
    DEFAULT_LOG_COLUMNS = VLMSurrogateDataset.DEFAULT_LOG_COLUMNS
    DEFAULT_EXPONENTIAL = VLMSurrogateDataset.DEFAULT_EXPONENTIAL

    def __init__(self, train_csv_path, test_csv_path, targets, **kwargs):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.targets = targets
        self.features = VLMSurrogateDataset.DEFAULT_FEATURES
        self.apply_log_transform = True

    def get_all_data(self, additional_csv_path=None):
        X = np.ones((4, len(self.features)))
        y = np.arange(4, dtype=float)
        return X, y


class FakeDivergenceModel:
    def predict_proba(self, X):
        X = np.asarray(X)
        total_samples = X[:, 9]
        return np.where(total_samples > np.log(1_000_000), 0.2, 0.8)


def make_vlm_config(total_samples_planned=55_000_000):
    return VLMConfig(
        lr=1e-4,
        wd=0.01,
        beta1=0.9,
        beta2=0.98,
        warmup_fraction=0.05,
        eps=1e-8,
        vision_width=256,
        text_width=256,
        total_samples_planned=total_samples_planned,
        training_progress=1.0,
    )


def test_vlm_query_returns_predictions_for_non_divergent_config(monkeypatch):
    import scan_benchmark.base_performance_benchmark as base_benchmark
    import scan_benchmark.vlm.api as vlm_api

    monkeypatch.setattr(base_benchmark, "TabPFNModel", FakePerformanceModel)
    monkeypatch.setattr(vlm_api, "VLMSurrogateDataset", FakeVLMDataset)
    monkeypatch.setattr(vlm_api, "BinaryBaggingEnsemble", lambda *args, **kwargs: FakeDivergenceModel())

    bench = vlm_api.VLMBenchmark(target=VLMTarget.VAL_LOSS, device="cpu")

    result = bench.query(make_vlm_config())

    assert result["failed"] is False
    assert result["divergence_probability"] == 0.2
    assert result["predictions"] == {
        "mean": 0.432,
        "uncertainty": 0.111,
    }
    assert result["model_stats"]["params"] > 0
    assert result["model_stats"]["total_flops"] > 0


def test_vlm_query_many_skips_performance_prediction_for_failed_configs(monkeypatch):
    import scan_benchmark.base_performance_benchmark as base_benchmark
    import scan_benchmark.vlm.api as vlm_api

    monkeypatch.setattr(base_benchmark, "TabPFNModel", FakePerformanceModel)
    monkeypatch.setattr(vlm_api, "VLMSurrogateDataset", FakeVLMDataset)
    monkeypatch.setattr(vlm_api, "BinaryBaggingEnsemble", lambda *args, **kwargs: FakeDivergenceModel())

    bench = vlm_api.VLMBenchmark(target=VLMTarget.TEST_LOSS, device="cpu")

    results = bench.query_many(
        [
            make_vlm_config(total_samples_planned=55_000_000),
            make_vlm_config(total_samples_planned=600_000),
        ]
    )

    assert len(results) == 2
    assert results[0]["failed"] is False
    assert results[0]["predictions"]["mean"] == 0.432
    assert results[1]["failed"] is True
    assert results[1]["predictions"] is None
    assert results[1]["divergence_probability"] == 0.8
