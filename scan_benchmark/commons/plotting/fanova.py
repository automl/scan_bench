from importlib.resources import files

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from fanova import fANOVA
from scan_benchmark.vlm.performance_surrogate.data import VLMSurrogateDataset

if __name__ == "__main__":
    train_path = files("scan_benchmark.vlm.performance_surrogate").joinpath("splits/train.csv")
    test_path = files("scan_benchmark.vlm.performance_surrogate").joinpath("splits/test.csv")

    dataset = VLMSurrogateDataset(
        train_csv_path=str(train_path),
        test_csv_path=str(test_path),
        targets=["val_loss"],
        seed=42,
        include_intermediate_points=True,
        eval_on_intermediate_points=True
    )

    X, y = dataset.get_all_data()

    cs = ConfigurationSpace()

    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.integer):
            cs.add_hyperparameter(
                UniformIntegerHyperparameter(
                    col,
                    lower=int(X[col].min()),
                    upper=int(X[col].max())
                )
            )
        else:
            cs.add_hyperparameter(
                UniformFloatHyperparameter(
                    col,
                    lower=float(X[col].min()),
                    upper=float(X[col].max())
                )
            )

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=0,
        n_jobs=-1
    )
    rf.fit(X.values, y)

    fanova = fANOVA(X.values, y, config_space=cs)

    print("\n=== Individual Feature Importance ===")
    for i, hp in enumerate(cs.get_hyperparameters()):
        importance = fanova.quantify_importance([i])[(i,)]['individual importance']
        print(f"{hp.name}: {importance:.4f}")

