import sys
import types
from enum import Enum
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest


class _StubTabPFNRegressor:
    def __init__(self, *args, **kwargs):
        self.fit_calls = []

    @classmethod
    def create_default_for_version(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def fit(self, X, y):
        self.fit_calls.append((np.asarray(X), np.asarray(y)))

    def predict(self, X, output_type=None):
        X = np.asarray(X)
        if output_type == "main":
            n = X.shape[0]
            return {
                "mean": np.full(n, 0.5),
                "quantiles": np.vstack([np.full(n, 0.4), np.full(n, 0.6)]),
            }
        return np.full(X.shape[0], 0.5)

class _StubModelVersion(Enum):
    V2_5 = "v2.5"


class _StubXGBClassifier:
    def __init__(self, *args, **kwargs):
        pass

    def set_params(self, **kwargs):
        return self

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(np.asarray(X).shape[0], dtype=int)

    def predict_proba(self, X):
        n = np.asarray(X).shape[0]
        return np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])

    def load_model(self, model_file):
        return None

    def save_model(self, model_file):
        return None


tabpfn_stub = types.ModuleType("tabpfn")
tabpfn_stub.TabPFNRegressor = _StubTabPFNRegressor
tabpfn_constants_stub = types.ModuleType("tabpfn.constants")
tabpfn_constants_stub.ModelVersion = _StubModelVersion
sys.modules.setdefault("tabpfn", tabpfn_stub)
sys.modules.setdefault("tabpfn.constants", tabpfn_constants_stub)

xgboost_stub = types.ModuleType("xgboost")
xgboost_stub.XGBClassifier = _StubXGBClassifier
sys.modules.setdefault("xgboost", xgboost_stub)

torch_stub = types.ModuleType("torch")
torch_stub.Tensor = type("Tensor", (), {})
sys.modules.setdefault("torch", torch_stub)


@pytest.fixture
def repo_tmp_path():
    path = Path.cwd() / ".test_tmp" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path
