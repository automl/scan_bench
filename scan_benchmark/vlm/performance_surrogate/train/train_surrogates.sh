#!/bin/bash

SCRIPT="scan_benchmark.vlm.performance_surrogate.train.train"
SEEDS=(42 73 94)
ENSEMBLE_TYPES=("xgb" "lightgbm" "mix")
DEVICE="cpu"

export PYTHONPATH="C:\Users\Donat\Documents\Master\Thesis\vlm scaling laws repository;$PYTHONPATH"

# tabpfn
for SEED in "${SEEDS[@]}"; do
    python -m "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED" \
        --labels val_loss \
        --device "$DEVICE" \

    python -m "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED" \
        --labels val_loss \
        --device "$DEVICE" \
        --include_intermediate_points

    python -m "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED" \
        --labels val_loss \
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
            --device "$DEVICE" \
            --labels val_loss

        python -m "$SCRIPT" \
            --model ensemble \
            --ensemble_type "$ENSEMBLE_TYPE" \
            --seed "$SEED" \
            --labels val_loss \
            --device "$DEVICE" \
            --include_intermediate_points

        python -m "$SCRIPT" \
            --model ensemble \
            --ensemble_type "$ENSEMBLE_TYPE" \
            --seed "$SEED" \
            --labels val_loss \
            --include_intermediate_points \
            --device "$DEVICE" \
            --eval_on_intermediate_points
    done
done

python -m "$SCRIPT" --model autogluon --seed 42 --include_intermediate_points --eval_on_intermediate_points
