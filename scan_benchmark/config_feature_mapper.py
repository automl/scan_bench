import numpy as np

from scan_benchmark.base_config import BaseConfig


class ConfigFeatureMapper:
    def __init__(
            self,
            feature_order,
            apply_log=False,
            log_columns=None,
            exponential_columns=None,
    ):
        self.feature_order = feature_order
        self.apply_log = apply_log
        self.log_columns = set(log_columns or [])
        # exponential of base 2
        self.exponential_columns = set(exponential_columns or [])

    def _build_row(self, config: BaseConfig, feature_order: list[str]) -> np.ndarray:
        config_dict = config.to_dict()
        missing = [name for name in feature_order if name not in config_dict]
        if missing:
            raise ValueError(f"Missing features in config: {missing}")

        row = []
        for name in feature_order:
            value = config_dict[name]

            if name in self.exponential_columns:
                value = 2 ** value

            elif self.apply_log and name in self.log_columns:
                value = np.log(float(value))

            row.append(value)

        return np.array(row, dtype=float).reshape(1, -1)

    def to_features(self, config: BaseConfig) -> np.ndarray:
        return self._build_row(config, self.feature_order)
