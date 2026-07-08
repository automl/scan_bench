import numpy as np
import pandas as pd

from scan_benchmark.dataset import BaseSurrogateDataset


class LLMSurrogateDataset(BaseSurrogateDataset):
    DEFAULT_TARGETS = ["valid_loss"]

    DEFAULT_FEATURES = [
        "d_model", "n_layers", "n_heads", "weight_decay", "beta1", "beta2",
        "cooldown_steps", "lr", "global_batch_size",
        "final_step", "total_compute", "n_data", "n_param", "tokens_so_far", "flops_so_far", "eval_step",
    ]

    DEFAULT_LOG_COLUMNS = [
        "weight_decay", "lr",
        "final_step", "total_compute", "n_data", "n_param",
        "tokens_so_far", "flops_so_far", "eval_step"
    ]

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
        self.train_csv_path = train_csv_path
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
        df = self._normalize_summary_columns(df)
        return self._filter_intermediate_points(
            df,
            keep_intermediate=self.include_intermediate_points,
        )

    def _prepare_test_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.test_csv_path is None:
            return df
        df = self._normalize_summary_columns(df)
        return self._filter_intermediate_points(
            df,
            keep_intermediate=self.eval_on_intermediate_points,
        )

    def _normalize_summary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "global_batch_size" not in df.columns:
            batch_cols = ["micro_batch_size", "grad_accumulation_steps", "recommended_num_gpus"]
            if all(col in df.columns for col in batch_cols):
                df["global_batch_size"] = (
                    df["micro_batch_size"]
                    * df["grad_accumulation_steps"]
                    * df["recommended_num_gpus"]
                )

        if "final_step" not in df.columns:
            if "resume_step" in df.columns:
                df["final_step"] = df["resume_step"]
            elif "eval_step" in df.columns:
                df["final_step"] = df.groupby(self.config_id_col)["eval_step"].transform("max")

        if "final_step" in df.columns and "eval_step" in df.columns:
            fallback_final_step = df.groupby(self.config_id_col)["eval_step"].transform("max")
            df["final_step"] = df["final_step"].fillna(fallback_final_step)

        alias_columns = {
            "total_compute": "total_flops",
            "n_data": "n_tokens",
            "n_param": "total_params",
        }
        for target_col, source_col in alias_columns.items():
            if target_col not in df.columns and source_col in df.columns:
                df[target_col] = df[source_col]

        self._repair_progress_columns(df)
        return df

    def _repair_progress_columns(self, df: pd.DataFrame) -> None:
        progress = pd.Series(np.nan, index=df.index, dtype=float)
        if "eval_step" in df.columns and "final_step" in df.columns:
            eval_step = pd.to_numeric(df["eval_step"], errors="coerce")
            final_step = pd.to_numeric(df["final_step"], errors="coerce")
            valid_final_step = final_step > 0
            progress.loc[valid_final_step] = eval_step.loc[valid_final_step] / final_step.loc[valid_final_step]

        if "step_fraction" in df.columns:
            step_fraction = pd.to_numeric(df["step_fraction"], errors="coerce")
            progress = progress.fillna(step_fraction)

        progress = progress.clip(lower=0.0)

        self._repair_scaled_progress_column(
            df=df,
            column="tokens_so_far",
            budget_column="n_data",
            progress=progress,
        )
        self._repair_scaled_progress_column(
            df=df,
            column="flops_so_far",
            budget_column="total_compute",
            progress=progress,
        )

    def _repair_scaled_progress_column(
            self,
            df: pd.DataFrame,
            column: str,
            budget_column: str,
            progress: pd.Series,
    ) -> None:
        if budget_column not in df.columns:
            return

        values = (
            pd.to_numeric(df[column], errors="coerce")
            if column in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        budget = pd.to_numeric(df[budget_column], errors="coerce")
        replacement = budget * progress

        missing_values = ~np.isfinite(values)
        usable_replacements = np.isfinite(replacement)
        values.loc[missing_values & usable_replacements] = replacement.loc[missing_values & usable_replacements]
        df[column] = values

    def _filter_intermediate_points(
            self,
            df: pd.DataFrame,
            keep_intermediate: bool,
    ) -> pd.DataFrame:
        if keep_intermediate:
            return df

        if "eval_step" not in df.columns:
            raise ValueError("Column 'eval_step' not found.")
        if "final_step" not in df.columns:
            raise ValueError("Column 'final_step' not found.")

        return df[df["eval_step"] == df["final_step"]].copy()

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

    def _get_test_bins(self):
        if "flops_bin" not in self.test_df.columns:
            return None
        return self.test_df["flops_bin"].to_numpy()

    def _get_top_performing_configs_per_bin(
            self,
            top_fraction: float = 0.1,
    ):
        df = self.test_df.copy()
        selection_col = "test_loss" if "test_loss" in df.columns else self.targets[0]

        last_eval_df = (
            df.sort_values("eval_step")
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
