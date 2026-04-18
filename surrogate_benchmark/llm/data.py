import numpy as np
import pandas as pd

class SurrogateDataset:
    """
    Loads train/test CSVs and provides deterministic config based subsets.
    Returns numpy arrays.
    """

    # DEFAULT_TARGETS = ["valid_loss"]
    DEFAULT_TARGETS = ["test_loss"]
    # DEFAULT_FEATURES = [
    #     "d_model", "n_layers", "n_heads", "weight_decay", "beta1", "beta2",
    #     "warmup_steps", "cooldown_steps", "initial_lr", "global_batch_size",
    #     "final_step", "total_compute", "n_data", "n_param", "micro_batch_size",
    #     "current_lr", "tokens_so_far", "train_loss_so_far", "flops_so_far", "eval_step",
    # ]
    DEFAULT_FEATURES = [
        "d_model", "n_layers", "n_heads", "weight_decay", "beta1", "beta2",
        "warmup_steps", "cooldown_steps", "initial_lr", "global_batch_size",
        "final_step", "total_compute", "n_data", "n_param",
        "current_lr", "tokens_so_far", "flops_so_far", "eval_step",
    ]

    def __init__(
        self,
        train_csv_path: str,
        test_csv_path: str,
        features: list[str] | None = None,
        targets: list[str] | None = None,
        seed: int = 42,
        config_id_col: str = "config_id",
    ):
        self.features = features if features is not None else self.DEFAULT_FEATURES
        self.targets = targets if targets is not None else self.DEFAULT_TARGETS
        self.seed = int(seed)
        self.config_id_col = config_id_col

        self.train_df = pd.read_csv(train_csv_path)
        self.test_df = pd.read_csv(test_csv_path)

        configs = np.array(sorted(self.train_df[self.config_id_col].unique()))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(configs)
        self._configs = configs

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

    def get_default_sizes(self):
        total = len(self._configs)
        sizes = [10, 20, 50, 100, 200]
        sizes = [s for s in sizes if s <= total]
        if total not in sizes:
            sizes.append(total)
        return sizes