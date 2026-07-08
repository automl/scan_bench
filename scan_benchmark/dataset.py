from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseSurrogateDataset(ABC):
    DEFAULT_TARGETS = []
    DEFAULT_FEATURES = []
    DEFAULT_LOG_COLUMNS = []

    def __init__(
            self,
            train_csv_path: str,
            test_csv_path: str | None = None,
            features: list[str] | None = None,
            targets: list[str] | None = None,
            seed: int = 42,
            apply_log_transform: bool = True,
    ):
        self.features = features if features is not None else self.DEFAULT_FEATURES
        self.targets = targets if targets is not None else self.DEFAULT_TARGETS
        self.seed = int(seed)
        self.apply_log_transform = apply_log_transform

        self.train_df = pd.read_csv(train_csv_path)
        self.test_df = pd.read_csv(test_csv_path) if test_csv_path is not None else pd.DataFrame(
            columns=self.DEFAULT_FEATURES)

        self.train_df = self._prepare_train_df(self.train_df)
        self.test_df = self._prepare_test_df(self.test_df)

        if self.apply_log_transform:
            self._apply_log_transform()

    def _prepare_train_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _prepare_test_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _get_test_bins(self):
        return None

    def _apply_log_transform(self):
        for col in self.DEFAULT_LOG_COLUMNS:
            if col in self.train_df.columns:
                self.train_df[col] = self._safe_log_transform(self.train_df[col], col)
            if col in self.test_df.columns:
                self.test_df[col] = self._safe_log_transform(self.test_df[col], col)

    def _safe_log_transform(self, series: pd.Series, col: str) -> pd.Series:
        try:
            values = series.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column '{col}' contains non-numeric values before log transform.") from exc

        finite_mask = np.isfinite(values)
        if not finite_mask.all():
            bad_count = int((~finite_mask).sum())
            raise ValueError(f"Column '{col}' contains {bad_count} non-finite values before log transform.")

        negative_mask = values < 0
        if negative_mask.any():
            bad_count = int(negative_mask.sum())
            raise ValueError(f"Column '{col}' contains {bad_count} negative values before log transform.")

        return np.log(values.clip(lower=np.finfo(float).tiny))

    def get_test_data(self):
        X_test = self.test_df[self.features].to_numpy()
        y_test = self.test_df[self.targets].to_numpy()
        return X_test, y_test

    def get_train_data(self):
        X_train = self.train_df[self.features].to_numpy()
        y_train = self.train_df[self.targets].to_numpy()
        return X_train, y_train

    def get_all_data(self, additional_csv_path: str | Path | None = None):
        dfs = [self.train_df]

        if self.test_df is not None and not self.test_df.empty:
            dfs.append(self.test_df)

        if additional_csv_path is not None:
            additional_df = pd.read_csv(additional_csv_path)
            dfs.append(additional_df)

        all_df = pd.concat(dfs, axis=0, ignore_index=True)

        X_all = all_df[self.features].to_numpy()
        y_all = all_df[self.targets].to_numpy()

        return X_all, y_all

    @abstractmethod
    def get_train_subset_df(self, n_cfg: int) -> pd.DataFrame:
        pass

    def get_train_subset(self, n_cfg: int):
        subset = self.get_train_subset_df(n_cfg)
        X = subset[self.features].to_numpy()
        y = subset[self.targets].to_numpy()
        return X, y

    def get_default_sizes(self, step: int = 10, max_points: int | None = None):
        total = self._get_size_base()
        if max_points is not None:
            total = min(total, max_points)

        if total <= 0:
            return []

        sizes = list(range(2, total + 1, step))
        if not sizes or sizes[-1] != total:
            sizes.append(total)

        return sizes

    @abstractmethod
    def _get_size_base(self) -> int:
        pass

    @abstractmethod
    def _get_top_performing_configs_per_bin(self, top_fraction: int = 0.1) -> pd.DataFrame:
        pass
