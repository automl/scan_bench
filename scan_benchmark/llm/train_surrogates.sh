#!/bin/bash

SCRIPT="scan_benchmark.llm.train"
SEEDS=(42 73 94)
ENSEMBLE_TYPES=("xgb" "lightgbm" "mix")
LABEL="test_loss"

# tabpfn
for SEED in "${SEEDS[@]}"; do
    python -m "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED" \
        --labels "$LABEL"

    python -m "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED" \
        --labels "$LABEL" \
        --include_intermediate_points

    python -m "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED" \
        --labels "$LABEL" \
        --include_intermediate_points \
        --eval_on_intermediate_points
done

# ensembles
for SEED in "${SEEDS[@]}"; do
    for ENSEMBLE_TYPE in "${ENSEMBLE_TYPES[@]}"; do
        python -m "$SCRIPT" \
            --model ensemble \
            --ensemble_type "$ENSEMBLE_TYPE" \
            --seed "$SEED" \
            --labels "$LABEL"

        python -m "$SCRIPT" \
            --model ensemble \
            --ensemble_type "$ENSEMBLE_TYPE" \
            --seed "$SEED" \
            --labels "$LABEL" \
            --include_intermediate_points

        python -m "$SCRIPT" \
            --model ensemble \
            --ensemble_type "$ENSEMBLE_TYPE" \
            --seed "$SEED" \
            --labels "$LABEL" \
            --include_intermediate_points \
            --eval_on_intermediate_points
    done
done

python -m "$SCRIPT" --model autogluon --seed 42 --include_intermediate_points --eval_on_intermediate_points