import numpy as np
import pytest

from scan_bench.base_config import BaseConfig
from scan_bench.config_feature_mapper import ConfigFeatureMapper


class TinyConfig(BaseConfig):
    def to_dict(self):
        return {
            "lr": 1e-3,
            "width": 256,
            "missing_from_order": 1,
        }


def test_feature_mapper_orders_features_and_applies_log_transform():
    mapper = ConfigFeatureMapper(
        feature_order=["width", "lr"],
        apply_log=True,
        log_columns=["lr"],
    )

    row = mapper.to_features(TinyConfig())

    assert row.shape == (1, 2)
    np.testing.assert_allclose(row, [[256, np.log(1e-3)]])


def test_feature_mapper_reports_missing_features():
    mapper = ConfigFeatureMapper(feature_order=["width", "unknown_feature"])

    with pytest.raises(ValueError, match="Missing features"):
        mapper.to_features(TinyConfig())
