import argparse
import random
from pathlib import Path

import numpy as np
import torch

from surrogate_benchmark.predictors.autogluon import AutoGluonModel
from surrogate_benchmark.predictors.ensembles import BaggingEnsemble, EnsembleType
from surrogate_benchmark.predictors_core.base import MultiLabelSurrogateModel
from surrogate_benchmark.predictors_core.pfn import TabPFNModel
from surrogate_benchmark.tabpfn.performance_surrogate.data import SurrogateDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_out_path(args):
    if args.model == "ensemble":
        model_name = args.ensemble_type
    else:
        model_name = args.model

    intermediate_flag = "fit_no_intermediate"
    predict_on_itermediate = "pred_no_intermediate"
    path = Path(
        "../results") / "predictors" / model_name / f"seed_{args.seed}" / intermediate_flag / predict_on_itermediate

    if args.model == "autogluon":
        ag_flag = "manual" if args.use_manual_ag_settings else "auto"
        path = path / ag_flag

    return path


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_csv", default="splits/train.csv")
    p.add_argument("--test_csv", default="splits/test.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labels", nargs="+", default=["val/val_loss"])
    p.add_argument("--model", choices=["tabpfn", "ensemble", "autogluon"], default="tabpfn")
    p.add_argument("--ensemble_type", choices=["xgb", "lightgbm", "mix"], default="xgb")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ag_time_limit", type=int, default=4 * 60 * 60)
    p.add_argument("--use_manual_ag_settings", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    out_path = build_out_path(args)
    out_path.mkdir(parents=True, exist_ok=True)
    time_limit = None

    dataset = SurrogateDataset(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        targets=args.labels,
        seed=args.seed,
    )

    sizes = dataset.get_default_sizes()
    labels = args.labels

    if args.model == "tabpfn":
        model = MultiLabelSurrogateModel(
            labels=labels,
            model_factory=lambda label: TabPFNModel(device=args.device),
        )
        model_name = "tabpfn"

    elif args.model == "ensemble":
        model = MultiLabelSurrogateModel(
            labels=args.labels,
            model_factory=lambda label: BaggingEnsemble(
                EnsembleType(args.ensemble_type)
            ),
        )
        model_name = f"ensemble_{args.ensemble_type}"

    elif args.model == "autogluon":
        sizes = [sizes[-1]]
        time_limit = args.ag_time_limit

        ag_flag = "manual" if args.use_manual_ag_settings else "auto"

        model = MultiLabelSurrogateModel(
            labels=args.labels,
            model_factory=lambda label: AutoGluonModel(
                features=dataset.features,
                label=label,
                time_limit=time_limit,
                base_path=f"AutogluonModels/{ag_flag}/seed_{args.seed}/{label}"
            ),
        )
        model_name = "autogluon"

    else:
        raise ValueError(f"Unknown model: {args.model}")

    results = model.validate(dataset, sizes, out_path)
