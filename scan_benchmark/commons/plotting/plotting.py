import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

PREDICTOR_COLORS = {
    "tabpfn": "#4C72B0",
    "xgb": "#DD8452",
    "ensemble_xgb": "#DD8452",
    "ensemble_lightgbm": "#55A868",
    "ensemble_mix": "#C44E52",
    "lightgbm": "#55A868",
    "mix": "#C44E52",
    "autogluon": "#8172B3",
}

TRAIN_MODES = ["fit_with_intermediate", "fit_no_intermediate"]
EVAL_MODES = ["pred_no_intermediate", "pred_with_intermediate"]
METRICS = ["rmse", "mae", "mdae", "marpd", "r2", "r", "spearman"]
TARGETS = ["val_loss"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=str,
        required=True,
        help="Path to predictor result folder",
    )
    parser.add_argument(
        "--plot-root",
        type=str,
        required=True,
        help="Path where plots should be saved",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["val_loss"],
        help="List of targets (e.g. val_loss test_loss)",
    )
    parser.add_argument(
        "--min-cfg",
        type=int,
        default=50,
        help="Minimum number of configs to show on the x axis",
    )
    return parser.parse_args()


def get_per_mode_plot_path(base_dir, target, metric, fit_mode, eval_mode):
    return Path(base_dir) / "per_mode" / target / metric / fit_mode / f"{eval_mode}.png"


def get_with_vs_without_plot_path(base_dir, target, metric, predictor, eval_mode):
    return Path(base_dir) / "with_vs_without" / target / metric / predictor / f"{eval_mode}.png"


def load_results(root_dir, targets):
    root = Path(root_dir)
    data = {}

    for predictor_dir in root.iterdir():
        if not predictor_dir.is_dir():
            continue

        predictor = predictor_dir.name
        data.setdefault(predictor, {})

        for seed_dir in predictor_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            try:
                seed = int(seed_dir.name.split("_")[-1])
            except ValueError:
                print(f"Skipping seed folder with unexpected name: {seed_dir}")
                continue

            for fit_mode_dir in seed_dir.iterdir():
                if not fit_mode_dir.is_dir():
                    continue

                fit_mode = fit_mode_dir.name
                data[predictor].setdefault(fit_mode, {})

                for eval_mode_dir in fit_mode_dir.iterdir():
                    if not eval_mode_dir.is_dir():
                        continue

                    eval_mode = eval_mode_dir.name
                    data[predictor][fit_mode].setdefault(eval_mode, {})

                    for target in targets:
                        target_path = eval_mode_dir / f"{target}.json"
                        if not target_path.exists():
                            continue

                        with open(target_path, "r") as f:
                            results_json = json.load(f)

                        data[predictor][fit_mode][eval_mode].setdefault(target, {})
                        data[predictor][fit_mode][eval_mode][target][seed] = results_json

    return data


def extract_curve(results_json, metric):
    pairs = []

    for n_cfg, metric_values in results_json.items():
        if metric in metric_values:
            pairs.append((int(n_cfg), metric_values[metric]))

    if not pairs:
        return np.array([]), np.array([])

    pairs.sort(key=lambda x: x[0])
    xs, ys = zip(*pairs)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def aggregate_curves_over_seeds(seed_results, metric, min_cfg=50):
    values_per_config = {}

    for results_json in seed_results.values():
        xs, ys = extract_curve(results_json, metric)
        for x, y in zip(xs, ys):
            values_per_config.setdefault(x, []).append(y)

    if not values_per_config:
        return np.array([]), np.array([]), np.array([])

    xs = np.array(sorted(values_per_config.keys()), dtype=float)
    mean = np.array([np.mean(values_per_config[x]) for x in xs], dtype=float)
    std = np.array([np.std(values_per_config[x]) for x in xs], dtype=float)

    mask = xs >= min_cfg
    return xs[mask], mean[mask], std[mask]


def setup_ax(ax, title, ylabel):
    ax.set_xscale("log")
    custom_ticks = [50, 100, 200, 500, 700]
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels([str(t) for t in custom_ticks])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Number of fitted configs", fontsize=12)
    ax.set_ylabel(ylabel.upper(), fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_and_close(fig, save_path=None, show=False):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_metric_across_predictors(
        all_results,
        target,
        metric,
        fit_mode,
        eval_mode,
        save_path=None,
        min_cfg=50,
        show=False,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted_any = False

    for predictor in sorted(all_results.keys()):
        seed_results = (
            all_results.get(predictor, {})
            .get(fit_mode, {})
            .get(eval_mode, {})
            .get(target, {})
        )

        if not seed_results:
            continue

        xs, mean, std = aggregate_curves_over_seeds(seed_results, metric, min_cfg=min_cfg)
        if len(xs) == 0:
            continue

        color = PREDICTOR_COLORS.get(predictor, "#333333")
        ax.plot(xs, mean, label=predictor, color=color, linewidth=2.5, alpha=0.95)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(f"No data found for target={target}, metric={metric}, fit_mode={fit_mode}, eval_mode={eval_mode}")
        return

    setup_ax(
        ax,
        title=(
            f"{target} | {metric.upper()}\n"
            f"Fit mode: {fit_mode} | Eval mode: {eval_mode}"
        ),
        ylabel=metric,
    )

    ax.legend(title="Predictor", loc="best", frameon=True, fontsize=10, title_fontsize=10)
    save_and_close(fig, save_path=save_path, show=show)


def plot_metric_with_vs_without(
        all_results,
        predictor,
        target,
        metric,
        eval_mode,
        save_path=None,
        min_cfg=50,
        show=False,
):
    fig, ax = plt.subplots(figsize=(11, 6))

    linestyle_map = {
        "fit_with_intermediate": "-",
        "fit_no_intermediate": "--",
    }
    alpha_map = {
        "fit_with_intermediate": 0.14,
        "fit_no_intermediate": 0.07,
    }

    color = PREDICTOR_COLORS.get(predictor, "#333333")
    plotted_any = False

    for fit_mode in TRAIN_MODES:
        seed_results = (
            all_results.get(predictor, {})
            .get(fit_mode, {})
            .get(eval_mode, {})
            .get(target, {})
        )

        if not seed_results:
            continue

        xs, mean, std = aggregate_curves_over_seeds(seed_results, metric, min_cfg=min_cfg)
        if len(xs) == 0:
            continue

        ax.plot(
            xs,
            mean,
            color=color,
            linestyle=linestyle_map[fit_mode],
            linewidth=2.3,
            alpha=0.95,
        )
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=alpha_map[fit_mode])
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(
            f"No comparison data found for predictor={predictor}, target={target}, metric={metric}, eval_mode={eval_mode}"
        )
        return

    setup_ax(
        ax,
        title=(
            f"{predictor} | {target} | {metric.upper()}\n"
            f"Compare fit_with_intermediate vs fit_no_intermediate | Eval mode: {eval_mode}"
        ),
        ylabel=metric,
    )

    legend_handles = [
        Line2D([0], [0], color="black", lw=2.3, linestyle="-", label="fit_with_intermediate"),
        Line2D([0], [0], color="black", lw=2.3, linestyle="--", label="fit_no_intermediate"),
    ]
    ax.legend(handles=legend_handles, title="Training mode", loc="best", frameon=True, fontsize=9, title_fontsize=10)
    save_and_close(fig, save_path=save_path, show=show)


def main():
    args = parse_args()

    targets = args.targets
    results = load_results(args.results_root, targets)
    plot_root = Path(args.plot_root)

    for target in targets:
        for fit_mode in TRAIN_MODES:
            for eval_mode in EVAL_MODES:
                for metric in METRICS:
                    plot_metric_across_predictors(
                        all_results=results,
                        target=target,
                        metric=metric,
                        fit_mode=fit_mode,
                        eval_mode=eval_mode,
                        save_path=get_per_mode_plot_path(
                            plot_root, target, metric, fit_mode, eval_mode
                        ),
                        min_cfg=args.min_cfg,
                        show=False,
                    )

    for target in targets:
        for predictor in results.keys():
            for eval_mode in ["pred_no_intermediate"]:
                for metric in METRICS:
                    plot_metric_with_vs_without(
                        all_results=results,
                        predictor=predictor,
                        target=target,
                        metric=metric,
                        eval_mode=eval_mode,
                        save_path=get_with_vs_without_plot_path(
                            plot_root, target, metric, predictor, eval_mode
                        ),
                        min_cfg=args.min_cfg,
                        show=False,
                    )


if __name__ == "__main__":
    main()
