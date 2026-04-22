import numpy as np
import pandas as pd


class DivergenceDataset:
    DEFAULT_FEATURES = [
        "lr", "wd", "beta1", "beta2", "eps", "warmup_fraction",
        "vision_width", "text_width", "global_batch_size",
        "total_samples_planned"]

    DEFAULT_LOG_COLUMNS = [
        "lr", "wd", "eps", "total_samples_planned"
    ]

    def __init__(
            self,
            train_csv_path: str,
            test_csv_path: str,
            features: list[str] | None = None,
            seed: int = 42,
            config_id_col: str = "config_id",
            apply_log_transform: bool = True,
    ):
        self.features = features if features is not None else self.DEFAULT_FEATURES
        self.targets = "failed"
        self.seed = int(seed)
        self.config_id_col = config_id_col
        self.apply_log_transform = apply_log_transform

        self.train_df = pd.read_csv(train_csv_path)
        self.test_df = pd.read_csv(test_csv_path)

        if self.apply_log_transform:
            self._apply_log_transform()

    def _apply_log_transform(self):
        for col in self.DEFAULT_LOG_COLUMNS:
            if col in self.train_df.columns:
                self.train_df[col] = np.log(self.train_df[col])
            if col in self.test_df.columns:
                self.test_df[col] = np.log(self.test_df[col])

    def get_train_data(self):
        X = self.train_df[self.features].values
        y = self.train_df[self.targets].values
        return X, y

    def get_test_data(self):
        X = self.test_df[self.features].values
        y = self.test_df[self.targets].values
        return X, y

    def get_all_data(self):
        all_df = pd.concat([self.train_df, self.test_df], axis=0, ignore_index=True)

        X_all = all_df[self.features].to_numpy()
        y_all = all_df[self.targets].to_numpy()
        return X_all, y_all
