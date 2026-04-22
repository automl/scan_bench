TABPFN_SEARCH_SPACE = {
    "hp_space": {
        "lr": {
            "choices": [5e-05, 1.44e-04, 4.16e-04, 1.2e-03, 3.47e-03, 1e-02],
        },
        "effective_batch_size": {
            "choices": [32, 64],
        },
        "weight_decay": {
            "choices": [0.0, 1e-4, 1e-1],
        },
    },
    "scale_space": {
        "embedding_size": {
            "choices": [16, 32, 64],
        },
        "num_layers": {
            "choices": [1, 2, 4],
        },
        "max_features": {
            "choices": [2, 4, 8],
        },
        "num_datapoints_max": {
            "choices": [100, 200, 400],
        },
    },
}
