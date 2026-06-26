VLM_SEARCH_SPACE = {
    "hp_space": {
        "lr": {
            "type": "float",
            "lower": 1e-5,
            "upper": 5e-2,
            "log": True,
        },
        "wd": {
            "type": "float",
            "lower": 1e-6,
            "upper": 2e-1,
            "log": True,
        },
        "warmup_fraction": {
            "type": "float",
            "lower": 0.0,
            "upper": 0.75,
        },
        "beta1": {
            "type": "float",
            "lower": 0.9,
            "upper": 0.99,
        },
        "beta2": {
            "type": "float",
            "lower": 0.95,
            "upper": 0.999,
        },
        "eps": {
            "type": "float",
            "lower": 1e-8,
            "upper": 1e-6,
            "log": True,
        },
        "train_num_samples": {
            "type": "float",
            "lower": 600_000,
            "upper": 200_000_000,
            "log": True,
        },
    },
    "arch_space": {
        "vision_width": {
            "choices": [32, 64, 128, 192, 256, 320, 384, 448, 512, 768, 1024],
        },
        "text_width": {
            "choices": [32, 64, 128, 192, 256, 320, 384, 448, 512, 768, 1024],
        },
    },
}
