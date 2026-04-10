#!/bin/bash

SCRIPT="train.py"
SEEDS=(42 73 94)
ENSEMBLE_TYPES=("xgb" "lightgbm" "mix")

export PYTHONPATH="C:\Users\Donat\Documents\Master\Thesis\vlm scaling laws repository;$PYTHONPATH"

# tabpfn
for SEED in "${SEEDS[@]}"; do
    python "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED"
done

## ensembles
#for SEED in "${SEEDS[@]}"; do
#    for ENSEMBLE_TYPE in "${ENSEMBLE_TYPES[@]}"; do
#        python "$SCRIPT" \
#            --model ensemble \
#            --ensemble_type "$ENSEMBLE_TYPE" \
#            --seed "$SEED" \
#    done
#done
