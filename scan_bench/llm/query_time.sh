#!/bin/bash

SCRIPT="scan_bench.commons.query_time.query_time_llm"
N_SEEDS=3
N_EVALS=100
ENSEMBLE_TYPES=("ensemble_xgb" "ensemble_lightgbm" "ensemble_mix")


# tabpfn
python -m "$SCRIPT" \
    --predictor_type tabpfn \
    --n_evals "$N_EVALS" \
    --n_seeds "$N_SEEDS" \
    --warmup \
    --use_batch

python -m "$SCRIPT" \
    --predictor_type tabpfn \
    --n_evals "$N_EVALS" \
    --n_seeds "$N_SEEDS" \
    --warmup

python -m "$SCRIPT" \
    --predictor_type tabpfn \
    --n_evals "$N_EVALS" \
    --n_seeds "$N_SEEDS" \
    --use_batch

python -m "$SCRIPT" \
    --predictor_type tabpfn \
    --n_evals "$N_EVALS" \
    --n_seeds "$N_SEEDS"

for ENSEMBLE_TYPE in "${ENSEMBLE_TYPES[@]}"; do
    python -m "$SCRIPT" \
        --predictor_type "$ENSEMBLE_TYPE" \
        --n_evals "$N_EVALS" \
        --n_seeds "$N_SEEDS" \
        --warmup \
        --use_batch
    python -m "$SCRIPT" \
        --predictor_type "$ENSEMBLE_TYPE" \
        --n_evals "$N_EVALS" \
        --n_seeds "$N_SEEDS" \
        --warmup
    python -m "$SCRIPT" \
        --predictor_type "$ENSEMBLE_TYPE" \
        --n_evals "$N_EVALS" \
        --n_seeds "$N_SEEDS" \
        --use_batch
    python -m "$SCRIPT" \
        --predictor_type "$ENSEMBLE_TYPE" \
        --n_evals "$N_EVALS" \
        --n_seeds "$N_SEEDS" 
done

# Uncomment the following lines to predict with autogluon
# python -m "$SCRIPT" \
#     --predictor_type autogluon \
#     --n_evals "$N_EVALS" \
#     --n_seeds "$N_SEEDS" \
#     --warmup "$WARMUP" \
#     --use_batch "$BATCH"