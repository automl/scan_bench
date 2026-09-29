import numpy as np

from scan_benchmark.llm.config import LLMConfig, LLMTarget
from scan_benchmark.llm.data import LLMSurrogateDataset


class FakePerformanceModel:
    def __init__(self, device="auto"):
        self.device = device
        self.fit_calls = []

    def fit(self, X, y):
        self.fit_calls.append((np.asarray(X), np.asarray(y)))

    def predict_with_uncertainty(self, X):
        n = np.asarray(X).shape[0]
        return {
            "mean": np.arange(n, dtype=float) + 1.23456,
            "uncertainty": np.full(n, 0.09876),
        }


class FakeLLMDataset:
    DEFAULT_LOG_COLUMNS = LLMSurrogateDataset.DEFAULT_LOG_COLUMNS
    DEFAULT_EXPONENTIAL = LLMSurrogateDataset.DEFAULT_EXPONENTIAL

    def __init__(self, train_csv_path, test_csv_path, targets, **kwargs):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.targets = targets
        self.features = LLMSurrogateDataset.DEFAULT_FEATURES
        self.apply_log_transform = False

    def get_all_data(self, additional_csv_path=None):
        X = np.ones((3, len(self.features)))
        y = np.arange(3, dtype=float)
        return X, y


def make_llm_config(training_progress=1.0):
    return LLMConfig(
        d_model=512,
        n_layers=12,
        n_heads=8,
        lr=3e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        cooldown_steps=0.2,
        n_tokens=1_000_000_000,
        training_progress=training_progress,
    )


def test_llm_query_returns_predictions_and_model_stats(monkeypatch):
    import scan_benchmark.base_performance_benchmark as base_benchmark
    import scan_benchmark.llm.api as llm_api

    monkeypatch.setattr(base_benchmark, "TabPFNModel", FakePerformanceModel)
    monkeypatch.setattr(llm_api, "LLMSurrogateDataset", FakeLLMDataset)

    bench = llm_api.LLMBenchmark(
        target=LLMTarget.VAL_LOSS,
        device="cpu",
    )

    result = bench.query(make_llm_config())

    assert set(result) == {"predictions", "model_stats"}
    assert result["predictions"] == {
        "mean": 1.235,
        "uncertainty": 0.099,
    }
    assert result["model_stats"]["total_params"] > 0
    assert result["model_stats"]["total_training_flops"] > 0


def test_llm_query_many_returns_one_result_per_config(monkeypatch):
    import scan_benchmark.base_performance_benchmark as base_benchmark
    import scan_benchmark.llm.api as llm_api

    monkeypatch.setattr(base_benchmark, "TabPFNModel", FakePerformanceModel)
    monkeypatch.setattr(llm_api, "LLMSurrogateDataset", FakeLLMDataset)

    bench = llm_api.LLMBenchmark(target=LLMTarget.TEST_LOSS, device="cpu")

    results = bench.query_many([make_llm_config(0.5), make_llm_config(1.0)])

    assert len(results) == 2
    assert results[0]["predictions"]["mean"] == 1.235
    assert results[1]["predictions"]["mean"] == 2.235
    assert all(result["model_stats"]["total_params"] > 0 for result in results)
