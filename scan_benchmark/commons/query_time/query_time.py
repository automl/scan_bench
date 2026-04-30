import time

import numpy as np

from scan_benchmark.vlm.config import VLMConfig


def surrogate_runtime_random_search(
    surrogate,
    config_sampler,
    n_evals=1000,
    n_seeds=100,
    warmup=True,
):
    if warmup:
        rng = np.random.default_rng(0)
        surrogate.query(config_sampler(rng))

    times = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)

        start = time.perf_counter()

        for _ in range(n_evals):
            config = config_sampler(rng)
            surrogate.query(config)

        end = time.perf_counter()
        times.append(end - start)

    times = np.asarray(times)

    return {
        "mean_cpu_seconds": float(times.mean()),
        "std_cpu_seconds": float(times.std()),
        "min_cpu_seconds": float(times.min()),
        "max_cpu_seconds": float(times.max()),
        "seconds_per_query": float(times.mean() / n_evals),
    }


def sample_vlm_config(rng):
    return VLMConfig(
        lr=10 ** rng.uniform(-5, -2),
        wd=10 ** rng.uniform(-5, -1),
        beta1=rng.uniform(0.8, 0.99),
        beta2=rng.uniform(0.9, 0.999),
        eps=1e-8,
        warmup_fraction=rng.uniform(0.0, 0.1),
        vision_width=int(rng.choice([32, 64, 128, 256])),
        text_width=int(rng.choice([32, 64, 128, 256])),
        total_samples_planned=int(rng.integers(1e6, 1e8)),
        training_progress=1.0,
    )

