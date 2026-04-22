import argparse


def get_base_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--test_csv", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labels", nargs="+", default=None)

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
    p.add_argument("--use_manual_ag_settings", action="store_true")

    return p


def finalize_args(args, defaults: dict):
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args
