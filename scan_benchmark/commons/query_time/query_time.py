import json
import time

import numpy as np

from scan_benchmark.llm.api import LLMBenchmark
from scan_benchmark.llm.config import LLMTarget, LLMConfig
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
        surrogate.query(config_sampler(rng))

    seed_stats = []
    all_query_times = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)

        query_times = []

        if use_batch:
            configs = [config_sampler(rng) for _ in range(n_evals)]

            start = time.perf_counter()
            surrogate.query_many(configs)
            end = time.perf_counter()

            total_time = end - start
            per_query = total_time / n_evals

            query_times = [per_query] * n_evals
        else:
            for _ in range(n_evals):
                config = config_sampler(rng)

                t0 = time.perf_counter()
                surrogate.query(config)
                t1 = time.perf_counter()

                query_times.append(t1 - t0)

        query_times = np.asarray(query_times)
        all_query_times.append(query_times)

        seed_stats.append({
            "mean": float(query_times.mean()),
            "std": float(query_times.std()),
            "min": float(query_times.min()),
            "max": float(query_times.max()),
        })

    all_query_times = np.concatenate(all_query_times)

    return {
        "per_seed": seed_stats,
        "overall": {
            "mean": float(all_query_times.mean()),
            "std": float(all_query_times.std()),
            "min": float(all_query_times.min()),
            "max": float(all_query_times.max()),
        },
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
    vlm_bench = VLMBenchmark(targets=[VLMTarget.VAL_LOSS], device="cpu")

    vlm_seq_results = surrogate_runtime(
        vlm_bench,
        sample_vlm_config,
        use_batch=False,
    )

    llm_bench = LLMBenchmark(targets=[LLMTarget.VAL_LOSS], device="cpu")

    llm_seq_results = surrogate_runtime(
        llm_bench,
        sample_llm_config,
        use_batch=False,
    )

    results = {
        "vlm": vlm_seq_results,
        "llm": llm_seq_results,
    }

    with open("runtime_results.json", "w") as f:
        json.dump(results, f, indent=4)
