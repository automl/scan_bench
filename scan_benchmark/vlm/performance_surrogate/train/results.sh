#!/bin/bash

SCRIPT="train.py"
SEEDS=(42 73 94)
ENSEMBLE_TYPES=("xgb" "lightgbm" "mix")

export PYTHONPATH="C:\Users\Donat\Documents\Master\Thesis\vlm scaling laws repository;$PYTHONPATH"

# tabpfn
for SEED in "${SEEDS[@]}"; do
    python "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED" \
        --labels val_loss

    python "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED" \
        --labels val_loss \
        --include_intermediate_points

    python "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED" \
        --labels val_loss \
        --include_intermediate_points \
        --eval_on_intermediate_points
done

# ensembles
for SEED in "${SEEDS[@]}"; do
    for ENSEMBLE_TYPE in "${ENSEMBLE_TYPES[@]}"; do
        python "$SCRIPT" \
            --model ensemble \
            --ensemble_type "$ENSEMBLE_TYPE" \
            --seed "$SEED" \
            --labels val_loss

        python "$SCRIPT" \
            --model ensemble \
            --ensemble_type "$ENSEMBLE_TYPE" \
            --seed "$SEED" \
            --labels val_loss \
            --include_intermediate_points

        python "$SCRIPT" \
            --model ensemble \
            --ensemble_type "$ENSEMBLE_TYPE" \
            --seed "$SEED" \
            --labels val_loss \
            --include_intermediate_points \
            --eval_on_intermediate_points
    done
done
