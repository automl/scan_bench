#!/bin/bash

SCRIPT="scan_bench.llm.train"
SEEDS=(42 73 94)
ENSEMBLE_TYPES=("xgb" "lightgbm" "mix")
FOLDS=(1 2 3 4 5)
LABEL="test_loss"
DEVICE="cuda"

# tabpfn
for SEED in "${SEEDS[@]}"; do
    python -m "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED" \
        --folds "${FOLDS[@]}" \
        --device "$DEVICE" \
        --labels "$LABEL"

    python -m "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED" \
        --folds "${FOLDS[@]}" \
        --labels "$LABEL" \
        --device "$DEVICE" \
        --include_intermediate_points

    python -m "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED" \
        --folds "${FOLDS[@]}" \
        --labels "$LABEL" \
        --include_intermediate_points \
        --device "$DEVICE" \
        --eval_on_intermediate_points
done

# ensembles
for SEED in "${SEEDS[@]}"; do
    for ENSEMBLE_TYPE in "${ENSEMBLE_TYPES[@]}"; do
        python -m "$SCRIPT" \
            --model ensemble \
            --ensemble_type "$ENSEMBLE_TYPE" \
            --seed "$SEED" \
            --folds "${FOLDS[@]}" \
            --device "$DEVICE" \
            --labels "$LABEL"

        python -m "$SCRIPT" \
            --model ensemble \
            --ensemble_type "$ENSEMBLE_TYPE" \
            --seed "$SEED" \
            --folds "${FOLDS[@]}" \
            --labels "$LABEL" \
            --device "$DEVICE" \
            --include_intermediate_points

        python -m "$SCRIPT" \
            --model ensemble \
            --ensemble_type "$ENSEMBLE_TYPE" \
            --seed "$SEED" \
            --folds "${FOLDS[@]}" \
            --labels "$LABEL" \
            --device "$DEVICE" \
            --include_intermediate_points \
            --eval_on_intermediate_points
    done
done

# Uncomment the following lines to run autogluon training
# python -m "$SCRIPT" \
#     --model autogluon \
#     --seed 42 \
#     --folds "${FOLDS[@]}" \
#     --include_intermediate_points \
#     --eval_on_intermediate_points

python -m scan_benchmark.commons.surrogate_evaluation.main_performance