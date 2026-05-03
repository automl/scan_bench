import json
import time

import numpy as np

from scan_benchmark.llm.api import LLMBenchmark
from scan_benchmark.llm.config import LLMConfig, LLMTarget


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


def sample_llm_config(rng: np.random.Generator) -> LLMConfig:
    d_model_factor = rng.integers(4, 19)
    d_model = d_model_factor * 64
    valid_n_heads = [n for n in range(3, 13) if d_model % (n * 2) == 0]
    n_heads = rng.choice(valid_n_heads) * 2

    return LLMConfig(
        d_model=d_model,
        n_layers=int(rng.uniform(4, 24)),
        n_heads=n_heads,
        lr=sample_log_uniform(rng, 1.0e-5, 1.0e-2),
        weight_decay=sample_log_uniform(rng, 1.0e-3, 2.0e-1),
        beta1=sample_log_uniform(rng, 0.9, 0.99),
        beta2=sample_log_uniform(rng, 0.95, 0.999),
        cooldown_steps=float(rng.uniform(0.0, 0.3)),
        n_tokens=int(sample_log_uniform(rng, 2e8, 1.6e10)),
        training_progress=1.0,
    )


if __name__ == "__main__":
    llm_bench = LLMBenchmark(targets=[LLMTarget.VAL_LOSS], device="cpu")

    seq_results = surrogate_runtime(
        llm_bench,
        sample_llm_config,
        use_batch=False,
    )

    batch_results = surrogate_runtime(
        llm_bench,
        sample_llm_config,
        use_batch=True,
    )

    results = {
        "sequential": seq_results,
        "batch": batch_results,
    }

    with open("runtime_results.json", "w") as f:
        json.dump(results, f, indent=4)
