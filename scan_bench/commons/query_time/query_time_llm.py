import argparse
import json
import time

import numpy as np

from scan_bench.base_performance_benchmark import PerformancePredictorType
from scan_bench.llm.api import LLMBenchmark
from scan_bench.llm.config import LLMTarget, LLMConfig


def surrogate_runtime(
        surrogate,
        config_sampler,
        n_evals=100,
        n_seeds=3,
        warmup=True,
        use_batch=True,
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

def sample_llm_config(rng: np.random.Generator) -> LLMConfig:
    d_model_factor = rng.integers(4, 25)
    d_model = d_model_factor * 64
    valid_n_heads = [n for n in range(3, 20) if d_model % (n * 2) == 0]
    n_heads = rng.choice(valid_n_heads) * 2

    return LLMConfig(
        d_model=d_model,
        n_layers=int(rng.uniform(4, 30)),
        n_heads=n_heads,
        lr=sample_log_uniform(rng, 1.0e-5, 1.0e-2),
        weight_decay=sample_log_uniform(rng, 1.0e-3, 2.0e-1),
        beta1=sample_log_uniform(rng, 0.9, 0.99),
        beta2=sample_log_uniform(rng, 0.95, 0.999),
        cooldown_steps=float(rng.uniform(0.0, 0.3)),
        n_tokens=int(sample_log_uniform(rng, 2e8, 4e10)),
        training_progress=1.0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_batch", action="store_true", default=False, help="Whether to use batch queries for the surrogate.")
    parser.add_argument("--n_evals", type=int, default=100, help="Number of evaluations to perform.")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of random seeds to use for evaluation.")
    parser.add_argument("--warmup", action="store_true", default=True, help="Whether to perform a warmup query before timing.")
    parser.add_argument("--predictor_type", type=str, default="tabpfn", choices=["tabpfn", "ensemble_xgb", "ensemble_lightgbm", "ensemble_mix", "autogluon"], help="Type of surrogate predictor to use.")

    args = parser.parse_args()

    predictor_type = PerformancePredictorType[args.predictor_type.upper()]
    llm_bench = LLMBenchmark(device="cpu", predictor_type=predictor_type)

    llm_seq_results = surrogate_runtime(
        llm_bench,
        sample_llm_config,
        use_batch=args.use_batch,
        n_evals=args.n_evals,
        n_seeds=args.n_seeds,
        warmup=args.warmup
    )

    results = {
        "llm": llm_seq_results,
    }

    with open(f"runtime_{args.predictor_type}_{args.n_evals}_evals_{args.n_seeds}_seeds_warmup_{args.warmup}_batch_{args.use_batch}_results.json", "w") as f:
        json.dump(results, f, indent=4)