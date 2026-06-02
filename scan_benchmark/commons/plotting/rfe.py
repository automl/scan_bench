from importlib.resources import files

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.feature_selection import RFE

from scan_benchmark.commons.predictors.ensembles import xgb_ensemble
from scan_benchmark.vlm.config import VLMTarget
from scan_benchmark.vlm.performance_surrogate.data import VLMSurrogateDataset


class XGBEnsembleRFE(BaseEstimator, RegressorMixin):

    def __init__(self, ensemble_list):
        self.ensemble_list = ensemble_list

    def fit(self, X, y):
        self.models_ = []

        feature_importances = []

        for model in self.ensemble_list:
            cloned_model = clone(model)

            cloned_model.fit(X, y)

            self.models_.append(cloned_model)

            feature_importances.append(
                cloned_model.feature_importances_
            )

        self.feature_importances_ = np.mean(
            feature_importances,
            axis=0
        )

        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X)
            for model in self.models_
        ])

        return predictions.mean(axis=1)


if __name__ == "__main__":
    train_path = files("scan_benchmark.vlm.performance_surrogate").joinpath("splits/train.csv")
    test_path = files("scan_benchmark.vlm.performance_surrogate").joinpath("splits/test.csv")

    dataset = VLMSurrogateDataset(
        train_csv_path=str(train_path),
        test_csv_path=str(test_path),
        targets=[VLMTarget.VAL_LOSS],
        seed=42,
        include_intermediate_points=False,
    )

    X, y = dataset.get_all_data()

    features = [
        "lr",
        "wd",
        "beta1",
        "beta2",
        "eps",
        "warmup_fraction",
        "vision_width",
        "text_width",
        "global_batch_size",
        "total_samples_planned",
    ]

    target = VLMTarget.VAL_LOSS.value

    ensemble = XGBEnsembleRFE(
        ensemble_list=xgb_ensemble(seed=42)
    )

    rfe = RFE(
        estimator=ensemble,
        n_features_to_select=1,
    )

    rfe.fit(X, y)

    ranking_df = (
        pd.DataFrame({
            "feature": features,
            "rank": rfe.ranking_,
        })
        .sort_values("rank")
    )

    print(ranking_df)
