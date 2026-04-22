from dataclasses import dataclass, asdict
from enum import Enum

from surrogate_benchmark.base_config import BaseConfig
from surrogate_benchmark.tabpfn.spaces import TABPFN_SEARCH_SPACE


@dataclass
class TabPFNConfig(BaseConfig):
    total_cells: int
    effective_batch_size: int
    lr: float
    max_features: int
    embedding_size: int
    num_layers: int
    num_datapoints_max: int
    weight_decay: float

    def __post_init__(self):
        self._validate_against_search_space()

    def to_dict(self) -> dict:
        key_mapping = {
            "total_cells": "total_cells",
            "effective_batch_size": "config/effective_batch_size",
            "lr": "config/lr",
            "max_features": "config/max_features",
            "embedding_size": "config/model_config.embedding_size",
            "num_layers": "config/model_config.num_layers",
            "num_datapoints_max": "config/num_datapoints_max",
            "weight_decay": "config/weight_decay",
        }

        raw = asdict(self)
        return {
            key_mapping.get(k, k): v
            for k, v in raw.items()
        }

    def _validate_against_search_space(self):
        hp_space = TABPFN_SEARCH_SPACE["hp_space"]
        scale_space = TABPFN_SEARCH_SPACE["scale_space"]

        for name, cfg in {**hp_space, **scale_space}.items():
            if not hasattr(self, name):
                continue

            val = getattr(self, name)

            if "choices" in cfg and val not in cfg["choices"]:
                raise ValueError(f"{name}={val} not in allowed choices {cfg['choices']}")

class TabPFNTarget(str, Enum):
    VAL_LOSS = "val/val_loss"
    NLL = "real_data/nll"
    ROC_AUC = "real_data/roc_auc"

    @classmethod
    def all(cls):
        return [t.value for t in cls]
