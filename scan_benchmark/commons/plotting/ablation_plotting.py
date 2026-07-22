import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    baseline_rmse = 0.208
    baseline_spearman = 0.984

    results = {
        "Learning rate": {"rmse": 0.8364, "spearman": 0.6825},
        "Weight decay": {"rmse": 0.2122, "spearman": 0.9839},
        "Warmup fraction": {"rmse": 0.3490, "spearman": 0.9536},
        r"$\beta_1$": {"rmse": 0.2298, "spearman": 0.9806},
        r"$\beta_2$": {"rmse": 0.2098, "spearman": 0.9845},
        r"$\epsilon$": {"rmse": 0.2031, "spearman": 0.9854},
        "Train samples": {"rmse": 0.3736, "spearman": 0.9442},
        "Vision width": {"rmse": 0.4500, "spearman": 0.9156},
        "Text width": {"rmse": 0.3535, "spearman": 0.9552},
        "Global batch size": {"rmse": 0.2019, "spearman": 0.9859},
        "Training progress": {"rmse": 0.3914, "spearman": 0.9355},
        "Learning-rate ratio": {"rmse": 0.2076, "spearman": 0.9847},
    }

    df = pd.DataFrame(results).T.reset_index()
    df.columns = ["Feature", "RMSE", "Spearman"]
    df["Delta RMSE"] = df["RMSE"] - baseline_rmse
    df["Delta Spearman"] = baseline_spearman - df["Spearman"]

    df = df.sort_values("Delta RMSE", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    axes[0].barh(df["Feature"], df["Delta RMSE"])
    axes[0].axvline(0, linewidth=1)
    axes[0].set_xlabel(r"$\Delta$ RMSE", fontsize=14)

    axes[1].barh(df["Feature"], df["Delta Spearman"])
    axes[1].axvline(0, linewidth=1)
    axes[1].set_xlabel(r"$\Delta$ Spearman", fontsize=14)

    axes[0].tick_params(axis="both", labelsize=12)
    axes[1].tick_params(axis="both", labelsize=12)

    bar_color = "teal"

    axes[0].barh(df["Feature"], df["Delta RMSE"], color=bar_color)
    axes[1].barh(df["Feature"], df["Delta Spearman"], color=bar_color)

    plt.tight_layout()
    plt.savefig("feature_ablation_side_by_side.pdf", bbox_inches="tight")
    plt.show()
