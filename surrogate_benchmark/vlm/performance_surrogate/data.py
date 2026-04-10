import numpy as np
import pandas as pd


class SurrogateDataset:
    """
    Loads train/test CSVs and provides deterministic config based subsets.
    Returns numpy arrays.
    """

    DEFAULT_TARGETS = ["val_loss"]

    DEFAULT_FEATURES = [
        "lr", "wd", "beta1", "beta2", "eps", "warmup_fraction",
        "vision_width", "text_width", "global_batch_size",
        "total_samples_planned",
        "training_progress", "lr_ratio",
    ]

    DEFAULT_LOG_COLUMNS = [
        "lr", "wd", "eps", "total_samples_planned"
    ]

    def __init__(
            self,
            train_csv_path: str,
            test_csv_path: str,
            features: list[str] | None = None,
            targets: list[str] | None = None,
            seed: int = 42,
            config_id_col: str = "config_id",
            include_intermediate_points: bool = True,
            eval_on_intermediate_points: bool = False,
            epoch_col: str = "epoch",
            epochs_col: str = "total_epochs",
            apply_log_transform: bool = True,
    ):
        self.features = features if features is not None else self.DEFAULT_FEATURES
        self.targets = targets if targets is not None else self.DEFAULT_TARGETS
        self.seed = int(seed)
        self.config_id_col = config_id_col
        self.include_intermediate_points = include_intermediate_points
        self.eval_on_intermediate_points = eval_on_intermediate_points
        self.epoch_col = epoch_col
        self.epochs_col = epochs_col
        self.apply_log_transform = apply_log_transform

        self.train_df = pd.read_csv(train_csv_path)
        self.test_df = pd.read_csv(test_csv_path)

        self.train_df = self._filter_intermediate_points(
            self.train_df,
            keep_intermediate=self.include_intermediate_points,
        )
        self.test_df = self._filter_intermediate_points(
            self.test_df,
            keep_intermediate=self.eval_on_intermediate_points,
        )

        if self.apply_log_transform:
            self._apply_log_transform()

        configs = np.array(sorted(self.train_df[self.config_id_col].unique()))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(configs)
        self._configs = configs

    def _filter_intermediate_points(
            self,
            df: pd.DataFrame,
            keep_intermediate: bool,
    ) -> pd.DataFrame:
        if keep_intermediate:
            return df

        if self.epoch_col not in df.columns:
            raise ValueError(f"Column '{self.epoch_col}' not found.")
        if self.epochs_col not in df.columns:
            raise ValueError(f"Column '{self.epochs_col}' not found.")
        if "epoch_diverged" not in df.columns:
            raise ValueError("Column 'epoch_diverged' not found.")

        mask = (df[self.epoch_col] == df[self.epochs_col]) | (df["epoch_diverged"] == True)

        return df[mask].copy()

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

    def get_train_subset_df(self, n_cfg: int) -> pd.DataFrame:
        selected = set(self._configs[: int(n_cfg)])
        return self.train_df[self.train_df[self.config_id_col].isin(selected)]

    def get_train_subset(self, n_cfg: int):
        subset = self.get_train_subset_df(n_cfg)
        X = subset[self.features].to_numpy()
        y = subset[self.targets].to_numpy()
        return X, y

    def get_default_sizes(self, step: int = 10):
        total = len(self._configs)
        if total <= 0:
            return []

        sizes = list(range(2, total + 1, step))

        if sizes[-1] != total:
            sizes.append(total)

        return sizes
