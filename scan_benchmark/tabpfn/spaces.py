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
            "lower": 16,
            "upper": 256,
        },
    },
    "scale_space": {
        "total_cells": {
            "type": "int",
            "lower": 1048576,
            "upper": 1073741824,
        },
        "embedding_size": {
            "choices": [16, 32, 64, 128, 256],
        },
        "num_layers": {
            "choices": [1, 2, 4, 8, 16, 32],
        },
        "max_features": {
            "type": "int",
            "lower": 32,
            "upper": 128,
        },
        "num_datapoints_max": {
            "type": "int",
            "lower": 128,
            "upper": 512,
        },
    },
}
