import numpy as np
import pandas as pd

from scan_bench.dataset import BaseSurrogateDataset


class TabPFNSurrogateDataset(BaseSurrogateDataset):
    DEFAULT_TARGETS = ["val/val_loss"]

    DEFAULT_FEATURES = [
        "total_cells",
        "config/effective_batch_size",
        "config/lr",
        "config/max_features",
        "config/model_config.embedding_size",
        "config/model_config.num_layers",
        "config/num_datapoints_max",
    ]

    DEFAULT_LOG_COLUMNS = ["config/lr"]

    # Configs carry the same units as the CSV columns, so no feature is rescaled.
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
            apply_log_transform: bool = True,
    ):
        self.test_csv_path = test_csv_path
        self.config_id_col = config_id_col
        self.include_intermediate_points = include_intermediate_points
        self.eval_on_intermediate_points = eval_on_intermediate_points

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

        if "is_final" not in df.columns:
            raise ValueError("Column 'is_final' not found.")

        return df[df["is_final"].astype(bool)].copy()

    def get_train_subset_df(self, n_cfg: int) -> pd.DataFrame:
        selected = set(self._configs[: int(n_cfg)])
        return self.train_df[self.train_df[self.config_id_col].isin(selected)]

    def _get_size_base(self) -> int:
        return len(self._configs)

    def _get_test_bins(self):
        if "flops_bin" not in self.test_df.columns:
            return None
        return self.test_df["flops_bin"].to_numpy()

    def _get_top_performing_configs_per_bin(
            self,
            top_fraction: float = 0.1,
    ):
        df = self.test_df.copy()
        selection_col = "val/val_loss" if "val/val_loss" in df.columns else self.targets[0]

        last_eval_df = (
            df.sort_values("step")
            .groupby(self.config_id_col)
            .tail(1)
        )

        top_config_dfs = []
        for _, bin_df in last_eval_df.groupby("flops_bin", observed=False):
            n_top = max(1, int(np.ceil(len(bin_df) * top_fraction)))
            top_config_dfs.append(
                bin_df.sort_values(selection_col).head(n_top)[[self.config_id_col]]
            )

        top_configs = pd.concat(top_config_dfs, ignore_index=True)

        filtered_df = df[
            df[self.config_id_col].isin(top_configs[self.config_id_col])
        ]

        X_test = filtered_df[self.features].to_numpy()
        y_test = filtered_df[self.targets].to_numpy()

        return X_test, y_test, filtered_df["flops_bin"].to_numpy()
