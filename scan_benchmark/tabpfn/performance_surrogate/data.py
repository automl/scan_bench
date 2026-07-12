import numpy as np
import pandas as pd

from scan_benchmark.dataset import BaseSurrogateDataset


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

    DEFAULT_EXPONENTIAL = ["total_cells", "config/effective_batch_size", "config/max_features",
                           "config/model_config.embedding_size",
                           "config/model_config.num_layers", "config/num_datapoints_max"]

    def _prepare_train_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def get_train_subset_df(self, n_cfg: int) -> pd.DataFrame:
        return self.train_df.iloc[:int(n_cfg)]

    def _get_size_base(self) -> int:
        return len(self.train_df)

    def _get_test_bins(self):
        return self.test_df["flops_bin"].to_numpy()

    def _get_top_performing_configs_per_bin(
            self,
            top_fraction: float = 0.1,
    ):
        df = self.test_df.copy()

        top_configs = (
            df
            .groupby("flops_bin", group_keys=False)
            .apply(
                lambda x: x.sort_values("val/val_loss").head(
                    max(1, int(np.ceil(len(x) * top_fraction)))
                ),
                include_groups=False,
            )[["config_id"]]
            .reset_index(drop=True)
        )

        filtered_df = df[
            df["config_id"].isin(top_configs["config_id"])
        ]

        X_test = filtered_df[self.features].to_numpy()
        y_test = filtered_df[self.targets].to_numpy()

        return X_test, y_test, filtered_df["flops_bin"].to_numpy()
