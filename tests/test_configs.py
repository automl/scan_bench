import pytest

from scan_bench.commons.query_time.query_time import sample_llm_config, sample_vlm_config
from scan_bench.llm.config import LLMConfig, LLMTarget
from scan_bench.vlm.config import VLMConfig, VLMTarget


def make_llm_config(**overrides):
    params = {
        "d_model": 512,
        "n_layers": 12,
        "n_heads": 8,
        "lr": 3e-3,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "cooldown_steps": 0.2,
        "n_tokens": 1_000_000_000,
        "training_progress": 1.0,
    }
    params.update(overrides)
    return LLMConfig(**params)


def make_vlm_config(**overrides):
    params = {
        "lr": 1e-4,
        "wd": 0.01,
        "beta1": 0.9,
        "beta2": 0.98,
        "warmup_fraction": 0.05,
        "eps": 1e-8,
        "vision_width": 256,
        "text_width": 256,
        "total_samples_planned": 55_000_000,
        "training_progress": 1.0,
    }
    params.update(overrides)
    return VLMConfig(**params)


def test_llm_config_to_dict_contains_surrogate_features():
    cfg = make_llm_config(training_progress=0.5)
    data = cfg.to_dict()

    assert data["d_model"] == 512
    assert data["lr"] == cfg.lr
    assert data["global_batch_size"] == 64
    assert data["tokens_so_far"] == data["n_data"]
    assert data["flops_so_far"] == data["total_compute"]


def test_llm_config_model_stats_scale_with_training_progress():
    half = make_llm_config(training_progress=0.5).compute_model_stats()
    full = make_llm_config(training_progress=1.0).compute_model_stats()

    assert half["total_params"] == full["total_params"]
    assert half["total_training_flops"] == full["total_training_flops"] * 0.5


def test_llm_config_rejects_example_invalid_head_divisibility():
    with pytest.raises(ValueError, match="must be divisible"):
        make_llm_config(d_model=256, n_heads=10)


def test_llm_config_rejects_example_invalid_d_model():
    with pytest.raises(ValueError, match="d_model"):
        make_llm_config(d_model=258)


def test_llm_target_all_lists_api_targets():
    assert LLMTarget.all() == [
        "valid_loss",
        "test_loss",
    ]


def test_vlm_config_derives_batch_size_and_lr_ratio():
    cfg = make_vlm_config(
        total_samples_planned=600_000,
        training_progress=0.025,
        warmup_fraction=0.05,
    )

    assert cfg.global_batch_size == 512
    assert cfg.lr_ratio == pytest.approx(0.5)
    assert cfg.to_dict()["lr_ratio"] == pytest.approx(0.5)


def test_vlm_config_rejects_values_outside_search_space():
    with pytest.raises(ValueError, match="lr"):
        make_vlm_config(lr=1e-7)

    with pytest.raises(ValueError, match="vision_width"):
        make_vlm_config(vision_width=33)


def test_vlm_target_all_includes_upstream_and_downstream_targets():
    all_targets = VLMTarget.all()

    assert "val_loss" in all_targets
    assert "test_loss" in all_targets
    assert "imagenet1k_acc1" in all_targets
    assert "fairness_fairface_acc_race_avg" in all_targets


def test_runtime_samplers_return_valid_configs():
    import numpy as np

    rng = np.random.default_rng(0)

    assert isinstance(sample_llm_config(rng), LLMConfig)
    assert isinstance(sample_vlm_config(rng), VLMConfig)
