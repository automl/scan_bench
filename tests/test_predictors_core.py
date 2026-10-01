import numpy as np
import pytest

from scan_bench.commons.predictors_core.base import MultiLabelSurrogateModel
from scan_bench.commons.predictors_core.pfn import TabPFNModel


class MeanModel:
    def fit(self, X, y):
        self.mean = float(np.mean(y))

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.mean)


def test_multilabel_surrogate_requires_fit_before_predict():
    model = MultiLabelSurrogateModel(["a"], lambda label: MeanModel())

    with pytest.raises(RuntimeError, match="fitted"):
        model.predict([[1.0]])


def test_multilabel_surrogate_fits_one_model_per_label():
    model = MultiLabelSurrogateModel(["a", "b"], lambda label: MeanModel())

    model.fit(np.array([[1], [2], [3]]), np.array([[1, 10], [2, 20], [3, 30]]))
    predictions = model.predict(np.array([[99], [100]]))

    assert predictions.tolist() == [[2.0, 20.0], [2.0, 20.0]]
    assert model.get_predictor("a").mean == 2.0


def test_multilabel_surrogate_validates_target_shape():
    model = MultiLabelSurrogateModel(["a", "b"], lambda label: MeanModel())

    with pytest.raises(ValueError, match="Expected y shape"):
        model.fit(np.array([[1], [2]]), np.array([1, 2]))


def test_tabpfn_quantile_alignment_accepts_sample_first_layout():
    model = TabPFNModel(device="cpu")

    aligned = model._align_quantiles_by_sample(np.array([[0.1, 0.9], [0.2, 0.8]]), 2)

    np.testing.assert_allclose(aligned, [[0.1, 0.9], [0.2, 0.8]])


def test_tabpfn_quantile_alignment_transposes_quantile_first_layout():
    model = TabPFNModel(device="cpu")

    aligned = model._align_quantiles_by_sample(
        np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]),
        3,
    )

    np.testing.assert_allclose(aligned, [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])


def test_tabpfn_quantile_alignment_rejects_unmatched_shape():
    model = TabPFNModel(device="cpu")

    with pytest.raises(ValueError, match="Could not align"):
        model._align_quantiles_by_sample(np.ones((3, 4)), 2)
