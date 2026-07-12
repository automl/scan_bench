TABPFN_SEARCH_SPACE = {
    "hp_space": {
        "lr": {
            "type": "float",
            "lower": 0.0001,
            "upper": 0.05,
            "log": True,
        },
        "effective_batch_size": {
            "type": "int",
            "lower": 4,
            "upper": 8,
            "power_2": True
        },
    },
    "scale_space": {
        "total_cells": {
            "type": "int",
            "lower": 20,
            "upper": 35,
            "power_2": True
        },
        "embedding_size": {
            "choices": [4, 8, 16, 32, 64, 128, 256],
        },
        "num_layers": {
            "type": "int",
            "lower": 0,
            "upper": 5,
            "power_2": True
        },
        "max_features": {
            "type": "int",
            "lower": 5,
            "upper": 7,
            "power_2": True
        },
        "num_datapoints_max": {
            "type": "int",
            "lower": 7,
            "upper": 9,
            "power_2": True
        },
    },
}
