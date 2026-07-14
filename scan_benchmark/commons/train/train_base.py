import random
from pathlib import Path

import numpy as np
import torch

from scan_benchmark.commons.predictors.autogluon import AutoGluonModel
from scan_benchmark.commons.predictors.ensembles import BaggingEnsemble, EnsembleType
from scan_benchmark.commons.predictors_core.base import MultiLabelSurrogateModel
from scan_benchmark.commons.predictors_core.pfn import TabPFNModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_out_path(args):
    if args.model == "ensemble":
        model_name = f"ensemble_{args.ensemble_type}"
    else:
        model_name = args.model

    intermediate_flag = "fit_with_intermediate" if getattr(args, "include_intermediate_points",
                                                           False) else "fit_no_intermediate"
    predict_on_intermediate = "pred_with_intermediate" if getattr(args, "eval_on_intermediate_points",
                                                                  False) else "pred_no_intermediate"

    path = (
            Path("scan_benchmark/vlm/performance_surrogate/results")
            / "predictors"
            / model_name
            / f"seed_{args.seed}"
            / intermediate_flag
            / predict_on_intermediate
    )

    if args.model == "autogluon":
        ag_flag = "manual" if args.use_manual_ag_settings else "auto"
        path = path / ag_flag

    return path


def build_final_model_out_path(args, model_family):
    if args.model == "ensemble":
        model_name = f"ensemble_{args.ensemble_type}"
    else:
        model_name = args.model

    intermediate_flag = "fit_with_intermediate" if getattr(args, "include_intermediate_points",
                                                           False) else "fit_no_intermediate"
    predict_on_intermediate = "pred_with_intermediate" if getattr(args, "eval_on_intermediate_points",
                                                                  False) else "pred_no_intermediate"

    path = (
            Path("scan_benchmark") / model_family / "performance_surrogate" / "saved_models"
            / "predictors"
            / model_name
            / f"seed_{args.seed}"
            / intermediate_flag
            / predict_on_intermediate
    )

    if args.model == "autogluon":
        ag_flag = "manual" if args.use_manual_ag_settings else "auto"
        path = path / ag_flag

    return path


def build_model(args, dataset, final_model_out_path):
    if args.model == "tabpfn":
        return MultiLabelSurrogateModel(
            labels=args.labels,
            model_factory=lambda label: TabPFNModel(device=args.device),
        )

    if args.model == "ensemble":
        return MultiLabelSurrogateModel(
            labels=args.labels,
            model_factory=lambda label: BaggingEnsemble(
                EnsembleType(args.ensemble_type)
            ),
        )

    if args.model == "autogluon":
        return MultiLabelSurrogateModel(
            labels=args.labels,
            model_factory=lambda label: AutoGluonModel(
                label=label,
                time_limit=args.ag_time_limit,
                base_path=final_model_out_path,
                fold=args.fold,
            ),
        )

    raise ValueError(f"Unknown model: {args.model}")


def build_dataset_kwargs(args, supports_intermediate_points: bool):
    kwargs = {
        "train_csv_path": args.train_csv,
        "test_csv_path": args.test_csv,
        "targets": args.labels,
        "features": getattr(args, "features", None),
        "seed": args.seed,
    }

    if supports_intermediate_points:
        kwargs["include_intermediate_points"] = args.include_intermediate_points
        kwargs["eval_on_intermediate_points"] = args.eval_on_intermediate_points

    return kwargs


def run_benchmark(args, dataset_cls, supports_intermediate_points: bool = False, model_family="vlm"):
    set_seed(args.seed)

    out_path = build_out_path(args) if args.out_dir is None else Path(args.out_dir)
    if out_path.exists() and any(out_path.iterdir()):
        print(f"Skipping validation, output directory already exists: {out_path}")
        return
    out_path.mkdir(parents=True, exist_ok=True)

    final_model_out_path = build_final_model_out_path(args, model_family)
    final_model_out_path.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = build_dataset_kwargs(args, supports_intermediate_points)
    dataset = dataset_cls(**dataset_kwargs)

    sizes = dataset.get_default_sizes()
    if args.model == "autogluon":
        sizes = [sizes[-1]]

    model = build_model(args, dataset, final_model_out_path)
    results = model.validate(dataset, sizes, out_path, final_model_out_path, args.additional_runs_path)

    return results
