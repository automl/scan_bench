import numpy as np
import pandas as pd

from scan_benchmark.dataset import BaseSurrogateDataset


class LLMSurrogateDataset(BaseSurrogateDataset):
    DEFAULT_TARGETS = ["valid_loss"]

    DEFAULT_FEATURES = [
        "d_model", "n_layers", "n_heads", "weight_decay", "beta1", "beta2",
        "warmup_steps", "cooldown_steps", "initial_lr", "global_batch_size",
        "final_step", "total_compute", "n_data", "n_param",
        "current_lr", "tokens_so_far", "flops_so_far", "eval_step",
    ]

    DEFAULT_LOG_COLUMNS = []

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

    def _get_test_data(self):
        X_test = self.test_df[self.features].to_numpy()
        y_test = self.test_df[self.targets].to_numpy()
        return X_test, y_test

    def _get_train_data(self):
        X_train = self.train_df[self.features].to_numpy()
        y_train = self.train_df[self.targets].to_numpy()
        return X_train, y_train

    def _get_all_data(self):
        all_df = pd.concat([self.train_df, self.test_df], axis=0, ignore_index=True)
        X_all = all_df[self.features].to_numpy()
        y_all = all_df[self.targets].to_numpy()
        return X_all, y_all
    
    def get_train_subset_df(self, n_cfg: int) -> pd.DataFrame:
        selected = set(self._configs[: int(n_cfg)])
        return self.train_df[self.train_df[self.config_id_col].isin(selected)]

    def _get_size_base(self) -> int:
        return len(self._configs)
