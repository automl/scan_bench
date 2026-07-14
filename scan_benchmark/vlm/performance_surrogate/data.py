import numpy as np
import pandas as pd

from scan_benchmark.dataset import BaseSurrogateDataset


class VLMSurrogateDataset(BaseSurrogateDataset):
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
    DEFAULT_EXPONENTIAL = []

    def __init__(
            self,
            train_csv_path: str,
            test_csv_path: str | None = None,
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
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.config_id_col = config_id_col
        self.include_intermediate_points = include_intermediate_points
        self.eval_on_intermediate_points = eval_on_intermediate_points
        self.epoch_col = epoch_col
        self.epochs_col = epochs_col

        super().__init__(
            train_csv_path=train_csv_path,
            test_csv_path=test_csv_path,
            features=features,
            targets=targets,
            seed=seed,
            apply_log_transform=apply_log_transform,
        )

        configs = np.array(sorted(self.train_df[self.config_id_col].unique()))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(configs)
        self._configs = configs

    def _prepare_train_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._filter_intermediate_points(
            df,
            keep_intermediate=self.include_intermediate_points,
        )

    def _prepare_test_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.test_csv_path is None:
            return df
        return self._filter_intermediate_points(
            df,
            keep_intermediate=self.eval_on_intermediate_points,
        )

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

    def get_train_subset_df(self, n_cfg: int) -> pd.DataFrame:
        selected = set(self._configs[:int(n_cfg)])
        return self.train_df[self.train_df[self.config_id_col].isin(selected)]

    def _get_size_base(self) -> int:
        return len(self._configs)

    def _get_test_bins(self):
        return self.test_df["flops_bin"].to_numpy()

    def _get_top_performing_configs_per_bin(
            self,
            top_fraction: float = 0.1,
    ):
        df = self.test_df.copy()

        last_epoch_df = (
            df.sort_values("epoch")
            .groupby("config_id")
            .tail(1)
        )

        top_configs = (
            last_epoch_df
            .groupby("flops_bin", group_keys=False)
            .apply(
                lambda x: x.sort_values("val_loss").head(
                    max(1, int(np.ceil(len(x) * top_fraction)))
                )
            )[["config_id"]]
            .reset_index(drop=True)
        )

        filtered_df = df[
            df["config_id"].isin(top_configs["config_id"])
        ]

        X_test = filtered_df[self.features].to_numpy()
        y_test = filtered_df[self.targets].to_numpy()

        return X_test, y_test, filtered_df["flops_bin"].to_numpy()
