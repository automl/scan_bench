from importlib.resources import files

from scan_benchmark.base_performance_benchmark import BasePerformanceBenchmark
from scan_benchmark.llm.config import LLMConfig, LLMTarget
from scan_benchmark.llm.data import LLMSurrogateDataset


class LLMBenchmark(BasePerformanceBenchmark):
    TARGET_ENUM = LLMTarget
    def __init__(
        self,
        targets: list[str] | None = None,
        device: str = "auto",
    ):
        targets = self._normalize_targets(targets)
        train_path = files("scan_benchmark.llm").joinpath("splits/train.csv")
        test_path = files("scan_benchmark.llm").joinpath("splits/test.csv")

        dataset = LLMSurrogateDataset(
            train_csv_path=str(train_path),
            test_csv_path=str(test_path),
            targets=targets,
            seed=42,
            include_intermediate_points=True,
        )
        super().__init__(surrogate_dataset=dataset, device=device)

    def query(self, config: LLMConfig) -> dict:
        model_stats = config.compute_model_stats()

        return {
            "predictions": self._predict_performance(config),
            "model_stats": model_stats,
        }

    def query_many(self, configs: list[LLMConfig]) -> list[dict]:
        stats_list = [config.compute_model_stats() for config in configs]
        predictions_list = self._predict_performance_many(configs)

        return [
            {
                "predictions": predictions,
                "model_stats": stats,
            }
            for predictions, stats in zip(predictions_list, stats_list)
        ]


if __name__ == "__main__":
    bench = LLMBenchmark(targets=[LLMTarget.VAL_LOSS, LLMTarget.TEST_LOSS, LLMTarget.HELLASWAG_ACC, LLMTarget.ARC_CHALLENGE_ACC, LLMTarget.ARC_EASY_ACC, LLMTarget.COPA_ACC, LLMTarget.OPENBOOKQA_ACC, LLMTarget.PIQA_ACC])

    config1 = LLMConfig(
        d_model=512,
        n_layers=12,
        n_heads=8,
        lr=3e-3,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        cooldown_steps=0.2,
        n_tokens=1_000_000_000,
        training_progress=1.0,
    )
    config2 = LLMConfig(
        d_model=768,
        n_layers=16,
        n_heads=12,
        lr=1e-3,
        weight_decay=0.05,
        beta1=0.9,
        beta2=0.95,
        cooldown_steps=0.2,
        n_tokens=5_000_000_000,
        training_progress=0.5,
    )
    config3 = LLMConfig(
        d_model=768,
        n_layers=16,
        n_heads=12,
        lr=1e-3,
        weight_decay=0.05,
        beta1=0.9,
        beta2=0.95,
        cooldown_steps=0.2,
        n_tokens=5_000_000_000,
        training_progress=1,
    )
    # results = bench.query_many([config1, config2, config3, config4, config5])
    # print(results)

    for label, make_config in [
        ("Config 1 (512d, 12L, 1B tokens, 100% progress)", lambda: config1),
        ("Config 2 (768d, 16L, 5B tokens, 50% progress)",  lambda: config2),
        ("Config 3 (768d, 16L, 5B tokens, 100% progress)", lambda: config3),
        ("Config 4 (256d, 12L, n_heads=10 - invalid divisibility)", lambda: LLMConfig(
            d_model=256,
            n_layers=12,
            n_heads=10,
            lr=1e-2,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
            cooldown_steps=0.2,
            n_tokens=1_000_000_000,
            training_progress=1.0,
        )),
        ("Config 5 (258d - out of range)", lambda: LLMConfig(
            d_model=258,
            n_layers=12,
            n_heads=8,
            lr=3e-3,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
            cooldown_steps=0.2,
            n_tokens=1_000_000_000,
            training_progress=1.0,
        )),
    ]:
        print(f"\n{label}:")
        try:
            cfg = make_config()
            result = bench.query(cfg)
            print(f"  Predictions: {result['predictions']}")
            print(f"  Model Stats: {result['model_stats']}")
            print(f"  Total params: {result['model_stats']['total_params']:,}")
            print(f"  Total FLOPs:  {result['model_stats']['total_training_flops_formatted']}")
        except ValueError as e:
            print(f"  [ERROR] Invalid config: {e}")


