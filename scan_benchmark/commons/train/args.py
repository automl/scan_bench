import argparse


def get_base_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--test_csv", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Input features used by the surrogate. Uses dataset defaults when omitted.",
    )
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--model_family", type=str, default="vlm",
                   help="Model family for which the surrogate is being fitted: vlm, llm, or tabpfn")

    p.add_argument(
        "--additional_runs_path",
        type=str,
        default=None,
        help="Path to a CSV containing additional training runs, which will be fitted to the final saved model.",
    )

    p.add_argument(
        "--model",
        choices=["tabpfn", "ensemble", "autogluon"],
        default="tabpfn",
    )
    p.add_argument(
        "--ensemble_type",
        choices=["xgb", "lightgbm", "mix"],
        default="xgb",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ag_time_limit", type=int, default=4 * 60 * 60)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--use_manual_ag_settings", action="store_true")

    return p


def finalize_args(args, defaults: dict):
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args
