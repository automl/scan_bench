import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseSurrogateDataset(ABC):
    DEFAULT_TARGETS = []
    DEFAULT_FEATURES = []
    DEFAULT_LOG_COLUMNS = []

    def __init__(
        self,
        train_csv_path: str,
        test_csv_path: str,
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
        self.test_df = pd.read_csv(test_csv_path)

        self.train_df = self._prepare_train_df(self.train_df)
        self.test_df = self._prepare_test_df(self.test_df)

        if self.apply_log_transform:
            self._apply_log_transform()

    def _prepare_train_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _prepare_test_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _apply_log_transform(self):
        for col in self.DEFAULT_LOG_COLUMNS:
            if col in self.train_df.columns:
                self.train_df[col] = np.log(self.train_df[col])
            if col in self.test_df.columns:
                self.test_df[col] = np.log(self.test_df[col])

    def get_test_data(self):
        X_test = self.test_df[self.features].to_numpy()
        y_test = self.test_df[self.targets].to_numpy()
        return X_test, y_test

    def get_train_data(self):
        X_train = self.train_df[self.features].to_numpy()
        y_train = self.train_df[self.targets].to_numpy()
        return X_train, y_train

    def get_all_data(self):
        all_df = pd.concat([self.train_df, self.test_df], axis=0, ignore_index=True)
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