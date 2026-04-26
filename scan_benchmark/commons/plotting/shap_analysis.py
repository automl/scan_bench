import pickle
from importlib.resources import files
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from tabpfn_extensions import interpretability

from scan_benchmark.commons.predictors_core.pfn import TabPFNModel
from scan_benchmark.vlm.performance_surrogate.data import VLMSurrogateDataset


def plot_shap(shap_values: np.ndarray) -> None:
    import shap

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]

    # Bar plot
    shap.plots.bar(shap_values=shap_values, show=False)
    plt.title("ScAn Parameters Global Importance (SHAP)")
    plt.savefig("shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Beeswarm
    shap.summary_plot(shap_values=shap_values, show=False)
    plt.title("Effect of ScAn Parameters on Validation Loss (SHAP)")
    plt.savefig("shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Interaction
    most_important = shap_values.abs.mean(0).values.argsort()[-1]
    if len(shap_values) > 1:
        plot_shap_feature(shap_values, most_important)


def plot_shap_feature(
        shap_values_: Any,
        feature_name: int | str,
        n_plots: int = 1,
) -> None:
    """Plot feature interactions for a specific feature based on SHAP values.

    Args:
        shap_values_: SHAP values object containing the data to plot.
        feature_name: The feature index or name to plot interactions for.
        n_plots: Number of interaction plots to create. Defaults to 1.

    Returns:
        None: This function only produces visualizations.
    """
    import shap

    # we can use shap.approximate_interactions to guess which features
    # may interact with age
    inds = shap.utils.potential_interactions(
        shap_values_[:, feature_name],
        shap_values_,
    )

    # make plots colored by each of the top three possible interacting features
    for i in range(n_plots):
        shap.plots.scatter(
            shap_values_[:, feature_name],
            color=shap_values_[:, inds[i]],
            show=False,
        )
        plt.savefig(f"shap_interaction_{feature_name}_{inds[i]}.png",
                    dpi=300, bbox_inches="tight")
        plt.show()


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

    performance_surrogate = TabPFNModel(device="cuda")
    X, y = dataset.get_all_data()
    X_test, _ = dataset.get_test_data()
    performance_surrogate.fit(X, y)

    shap_values = interpretability.shap.get_shap_values(
        estimator=performance_surrogate,
        test_x=X_test,
        attribute_names=dataset.DEFAULT_FEATURES,
        algorithm="permutation",
    )

    with open("shap_values.pkl", "wb") as f:
        pickle.dump(shap_values, f)

    plot_shap(shap_values)
