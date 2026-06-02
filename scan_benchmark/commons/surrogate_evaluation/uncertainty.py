import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr


def load_uncertainty_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    y_true = np.array(data["y_true"])
    y_pred = np.array(data["mean"])
    quantiles = np.array(data["quantiles"])

    p10 = quantiles[:, 0]
    p90 = quantiles[:, 8]

    uncertainty_width = p90 - p10
    abs_error = np.abs(y_true - y_pred)

    return uncertainty_width, abs_error


def compute_spearman_uncertainty_error(json_path):
    uncertainty_width, abs_error = load_uncertainty_data(json_path)

    rho, p_value = spearmanr(uncertainty_width, abs_error)

    print(f"Spearman rho = {rho:.4f}")
    print(f"p-value = {p_value:.4e}")
    print(f"Mean uncertainty width = {uncertainty_width.mean():.4f}")
    print(f"Mean absolute error = {abs_error.mean():.4f}")

    return rho, p_value


def keep_only_nonfailed_last_epoch(df: pd.DataFrame) -> pd.DataFrame:
    df_last = (
        df
        .sort_values(["config_id", "epoch"])
        .drop_duplicates(subset="config_id", keep="last")
        .reset_index(drop=True)
    )

    return df_last


def plot_uncertainty_error_flops(
        json_path,
        test_csv_path,
        flops_col="Total compute(GLOPs)",
        x_axis="uncertainty",  # "uncertainty" or "flops"
        color_by="flops",  # "flops" or "uncertainty"
):
    uncertainty_width, abs_error = load_uncertainty_data(json_path)

    test_df = pd.read_csv(test_csv_path)
    test_df = keep_only_nonfailed_last_epoch(test_df)

    assert len(test_df) == len(uncertainty_width)

    df = pd.DataFrame({
        "uncertainty_width": uncertainty_width,
        "abs_error": abs_error,
        "flops": test_df[flops_col].values,
    })

    if x_axis == "uncertainty":
        x = df["uncertainty_width"]
        x_label = "Uncertainty width (p90 - p10)"
        use_log_x = True

    elif x_axis == "flops":
        x = df["flops"]
        x_label = "GFLOPs"
        use_log_x = True

    if color_by == "flops":
        colors = np.log10(df["flops"])
        color_label = "GFLOPs"

    elif color_by == "uncertainty":
        colors = df["uncertainty_width"]
        color_label = "Uncertainty width (p90 - p10)"

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        x,
        df["abs_error"],
        c=colors,
        cmap="viridis",
        alpha=0.7,
        s=20,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label(color_label)

    if use_log_x:
        plt.xscale("log")

    plt.xlabel(x_label)
    plt.ylabel("Absolute prediction error")

    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.show()


def check_uncertainty_calibration(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    y_true = np.array(data["y_true"])
    quantiles = np.array(data["quantiles"])

    intervals = [
        (0, 8, 0.80, "P10-P90"),
        (1, 7, 0.60, "P20-P80"),
        (2, 6, 0.40, "P30-P70"),
        (3, 5, 0.20, "P40-P60"),
    ]

    results = []

    for low_idx, high_idx, expected_coverage, name in intervals:
        lower = quantiles[:, low_idx]
        upper = quantiles[:, high_idx]

        inside = (y_true >= lower) & (y_true <= upper)
        observed_coverage = inside.mean()

        results.append({
            "interval": name,
            "expected_coverage": expected_coverage,
            "observed_coverage": observed_coverage,
            "calibration_error": abs(observed_coverage - expected_coverage),
        })

        print(
            f"{name}: expected={expected_coverage:.1%}, "
            f"observed={observed_coverage:.1%}, "
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    json_path = "../../vlm/performance_surrogate/results/predictors/tabpfn/seed_42/fit_no_intermediate/pred_no_intermediate/full_pred.json"
    test_csv_path = "../../vlm/performance_surrogate/splits/test.csv"
    compute_spearman_uncertainty_error(json_path)
    plot_uncertainty_error_flops(
        json_path,
        test_csv_path,
        x_axis="uncertainty",
        color_by="flops",
    )
    plot_uncertainty_error_flops(
        json_path,
        test_csv_path,
        x_axis="flops",
        color_by="uncertainty",
    )
    check_uncertainty_calibration(json_path)
