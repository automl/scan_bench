import numpy as np
import pytest

from scan_bench.commons.metrics.metrics import (
    calculate_correlation_matrix,
    compute_regression_metrics,
    mean_absolute_percentage_relative_deviation,
)


def test_mean_absolute_percentage_relative_deviation_handles_zero_denominator():
    value = mean_absolute_percentage_relative_deviation([0, 2], [0, 4])

    assert value == pytest.approx(33.3333333333)


def test_compute_regression_metrics_for_perfect_predictions():
    metrics = compute_regression_metrics([1, 2, 3, 4], [1, 2, 3, 4])

    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert metrics["mdae"] == 0
    assert metrics["marpd"] == 0
    assert metrics["r2"] == 1
    assert metrics["r"] == pytest.approx(1)
    assert metrics["spearman"] == pytest.approx(1)


def test_calculate_correlation_matrix_returns_kendall_matrix_and_keys():
    corr, keys = calculate_correlation_matrix(
        {
            "ascending": [1, 2, 3],
            "descending": [3, 2, 1],
        }
    )

    assert keys == ["ascending", "descending"]
    np.testing.assert_allclose(corr, [[1.0, -1.0], [-1.0, 1.0]])
