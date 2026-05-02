import json
import time

import numpy as np

from scan_benchmark.vlm.api import VLMBenchmark
from scan_benchmark.vlm.config import VLMConfig, VLMTarget

VISION_TEXT_WIDTHS = [32, 64, 128, 192, 256, 320, 384, 448, 512]


def surrogate_runtime(
        surrogate,
        config_sampler,
        n_evals=2,
        n_seeds=1,
        warmup=True,
        use_batch=False,
):
    if warmup:
        rng = np.random.default_rng(0)
        if use_batch:
            surrogate.query_many([config_sampler(rng) for _ in range(10)])
        else:
            surrogate.query(config_sampler(rng))

    times = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)

        start = time.perf_counter()

        if use_batch:
            # batch mode
            configs = [config_sampler(rng) for _ in range(n_evals)]
            surrogate.query_many(configs)
        else:
            # sequential mode
            for _ in range(n_evals):
                config = config_sampler(rng)
                surrogate.query(config)

        end = time.perf_counter()
        times.append(end - start)

    times = np.asarray(times)

    return {
        "mode": "batch" if use_batch else "sequential",
        "search_time_seconds": {
            "mean": float(times.mean()),
            "std": float(times.std()),
            "min": float(times.min()),
            "max": float(times.max()),
        },
        "time_per_query_seconds": float(times.mean() / n_evals),
        "n_evals": n_evals,
        "n_seeds": n_seeds,
    }


def sample_log_uniform(rng, lower, upper):
    return float(np.exp(rng.uniform(np.log(lower), np.log(upper))))


def sample_vlm_config(rng: np.random.Generator) -> VLMConfig:
    train_num_samples_million = sample_log_uniform(rng, 0.6, 60.0)
    # querying the surrogate is agnostic to the sampling strategy
    return VLMConfig(
        lr=sample_log_uniform(rng, 1.0e-5, 5.0e-2),
        wd=sample_log_uniform(rng, 1.0e-6, 2.0e-1),
        beta1=sample_log_uniform(rng, 0.9, 0.99),
        beta2=sample_log_uniform(rng, 0.95, 0.999),
        eps=sample_log_uniform(rng, 1.0e-8, 1.0e-6),
        warmup_fraction=float(rng.uniform(0.0, 0.75)),
        vision_width=int(rng.choice(VISION_TEXT_WIDTHS)),
        text_width=int(rng.choice(VISION_TEXT_WIDTHS)),
        total_samples_planned=int(train_num_samples_million * 1_000_000),
        training_progress=1.0,
    )


if __name__ == "__main__":
    vlm_bench = VLMBenchmark(targets=[VLMTarget.VAL_LOSS], device="cpu")

    seq_results = surrogate_runtime(
        vlm_bench,
        sample_vlm_config,
        use_batch=False,
    )

    batch_results = surrogate_runtime(
        vlm_bench,
        sample_vlm_config,
        use_batch=True,
    )

    results = {
        "sequential": seq_results,
        "batch": batch_results,
    }

    with open("runtime_results.json", "w") as f:
        json.dump(results, f, indent=4)
