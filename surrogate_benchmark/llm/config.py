import math
from dataclasses import dataclass

import numpy as np

from surrogate_benchmark.llm.utils import (
    tot_params,
    non_embedding_params,
    flops_per_token_training,
    total_training_flops,
    format_flops,
    get_target_global_batch_for_tokens
)

@dataclass
class LLMConfig:
    """Configuration for LLM Benchmark.
    Fixed architecture constants:
        vocab_size = 50277,  seq_len = 2048,  mlp = SwiGLU (8/3),
        weight_tying = True,  lr_end = 0.0  (WSD schedule).
    """

    d_model: int
    n_layers: int
    n_heads: int
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    cooldown_steps: float
    n_tokens: int
    training_progress: float = 1.0
    warmup_steps: float = 0.05

    # Architecture constants
    VOCAB_SIZE: int = 50277
    SEQ_LEN: int = 2048
    MLP_CLASS: str = "glu"
    EXPAND: float = 8 / 3
    WEIGHT_TYING: bool = True
    LR_END: float = 0.0


    def __post_init__(self):
        valid_d_model = [i * 64 for i in range(4, 19)]
        valid_n_heads = [i * 2  for i in range(3, 13)]

        bounds = {
            "lr":             (1.0e-5, 1.0e-2),
            "weight_decay":   (1.0e-3, 0.2),
            "n_layers":       (4, 24),
            "beta1":          (0.8, 0.99),
            "beta2":          (0.9, 0.999),
            "cooldown_steps": (0.0, 0.3),
            "n_tokens":       (200_000_000, 16_000_000_000),
        }
        param_values = {
            "lr":             self.lr,
            "weight_decay":   self.weight_decay,
            "n_layers":       self.n_layers,
            "beta1":          self.beta1,
            "beta2":          self.beta2,
            "cooldown_steps": self.cooldown_steps,
            "n_tokens":       self.n_tokens,
        }

        for param, value in param_values.items():
            lo, hi = bounds[param]
            if not (lo <= value <= hi):
                raise ValueError(
                    f"Parameter '{param}' = {value} is outside the search space range "
                    f"[{lo}, {hi}]."
                )

        if self.d_model not in valid_d_model:
            raise ValueError(
                f"Parameter 'd_model' = {self.d_model} is outside the search space. "
                f"d_model must be a multiple of 64 in the range "
                f"[{valid_d_model[0]}, {valid_d_model[-1]}]. "
                f"Valid values: {valid_d_model}"
            )

        if self.n_heads not in valid_n_heads:
            raise ValueError(
                f"Parameter 'n_heads' = {self.n_heads} is outside the search space. "
                f"n_heads must be a multiple of 2 in the range "
                f"[{valid_n_heads[0]}, {valid_n_heads[-1]}]. "
                f"Valid values: {valid_n_heads}"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"'d_model' ({self.d_model}) must be divisible by 'n_heads' ({self.n_heads}). "
                f"Current ratio d_model / n_heads = {self.d_model / self.n_heads:.4f} is not an integer. "
                f"Choose n_heads that evenly divides d_model = {self.d_model}."
            )

    def _compute_current_lr(self, final_step: int) -> float:
        """Approximate current_lr at final_step using a WSD schedule."""
        warmup_abs = self.warmup_steps * final_step
        cooldown_abs = self.cooldown_steps * final_step
        cooldown_start = final_step - cooldown_abs

        t = final_step
        if t <= warmup_abs:
            return self.lr * t / max(warmup_abs, 1)
        if t <= cooldown_start:
            return self.lr
        if cooldown_abs <= 0:
            return self.LR_END
        progress = (t - cooldown_start) / cooldown_abs
        return self.lr + (self.LR_END - self.lr) * min(progress, 1.0)

    def compute_model_stats(self) -> dict:
        """Compute model parameter count and FLOPs cost."""
        n_params = tot_params(
            self.n_layers,
            self.d_model,
            self.VOCAB_SIZE,
            self.WEIGHT_TYING,
            self.MLP_CLASS,
            self.EXPAND,
        )
        n_non_emb = non_embedding_params(
            self.n_layers,
            self.d_model,
            self.VOCAB_SIZE,
            self.WEIGHT_TYING,
            self.MLP_CLASS,
            self.EXPAND,
        )
        flops_per_tok = flops_per_token_training(
            self.n_layers, self.d_model, self.VOCAB_SIZE, self.SEQ_LEN, self.MLP_CLASS, self.EXPAND
        )
        # Adjust for training_progress
        effective_tokens = self.n_tokens * self.training_progress
        total_flops = total_training_flops(
            self.n_layers,
            self.d_model,
            self.VOCAB_SIZE,
            self.SEQ_LEN,
            effective_tokens,
            self.MLP_CLASS,
            self.EXPAND,
        )
        return {
            "total_params": n_params,
            "non_embedding_params": n_non_emb,
            "flops_per_token_training": flops_per_tok,
            "total_training_flops": total_flops,
            "total_training_flops_formatted": format_flops(total_flops),
        }

    def build_feature_row(self, features: list[str]) -> np.ndarray:
        """
        Construct the feature vector expected by the surrogate model.
        """
        effective_tokens = self.n_tokens * self.training_progress
        global_batch_size = get_target_global_batch_for_tokens(self.n_tokens)
        final_step = effective_tokens // (self.SEQ_LEN * global_batch_size)
        n_data = final_step * self.SEQ_LEN * global_batch_size

        n_param = tot_params(
            self.n_layers,
            self.d_model,
            self.VOCAB_SIZE,
            self.WEIGHT_TYING,
            self.MLP_CLASS,
            self.EXPAND,
        )
        total_flops = total_training_flops(
            self.n_layers,
            self.d_model,
            self.VOCAB_SIZE,
            self.SEQ_LEN,
            int(effective_tokens),
            self.MLP_CLASS,
            self.EXPAND,
        )
        current_lr = self._compute_current_lr(int(final_step))

        feature_map = {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "warmup_steps": self.warmup_steps,
            "cooldown_steps": self.cooldown_steps,
            "initial_lr": self.lr,
            "global_batch_size": global_batch_size,
            "final_step": final_step,
            "total_compute": total_flops,
            "n_data": n_data,
            "n_param": n_param,
            "current_lr": current_lr,
            "tokens_so_far": n_data,
            "flops_so_far": total_flops,
            "eval_step": final_step,
        }

        row = np.array(
            [[feature_map[f] for f in features]],
            dtype=np.float64,
        )
        return row