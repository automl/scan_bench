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

    DEFAULT_LOG_COLUMNS = ["total_cells", "config/lr"]

    def _prepare_train_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def get_train_subset_df(self, n_cfg: int) -> pd.DataFrame:
        return self.train_df.iloc[:int(n_cfg)]

    def _get_size_base(self) -> int:
        return len(self.train_df)

    def _get_test_bins(self):
        return self.test_df["flops_bin"].to_numpy()
