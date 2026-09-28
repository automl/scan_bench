# All values are the real quantities (layer counts, cells, feature counts), matching
# data.csv, flops.json and n_params.json. Base-2 grids belong to the optimizer's
# search space, not to TabPFNConfig.
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
            "upper": 34359738368,
        },
        "embedding_size": {
            "choices": [4, 8, 16, 32, 64, 128, 256],
        },
        "num_layers": {
            "type": "int",
            "lower": 2,
            "upper": 32,
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
