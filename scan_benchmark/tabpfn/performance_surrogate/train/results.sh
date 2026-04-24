#!/bin/bash
#SBATCH --job-name=surrogate_train
#SBATCH --output=surrogate_logs/%x_%j.out
#SBATCH --error=surrogate_logs/%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=mldlc2_cpu-epyc9655
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh && conda activate surrogate

export PYTHONPATH="/work/dlclarge1/sinanid-VLM-scaling-law/scan_bench_suite:$PYTHONPATH"

SCRIPT="train.py"
SEEDS=(42 73 94)
ENSEMBLE_TYPES=("xgb" "lightgbm" "mix")

# tabpfn
for SEED in "${SEEDS[@]}"; do
    python "$SCRIPT" \
        --model tabpfn \
        --seed "$SEED"
done

# ensembles
for SEED in "${SEEDS[@]}"; do
    for ENSEMBLE_TYPE in "${ENSEMBLE_TYPES[@]}"; do
        python "$SCRIPT" \
            --model ensemble \
            --ensemble_type "$ENSEMBLE_TYPE" \
            --seed "$SEED"
    done
done

python "$SCRIPT" --model autogluon --seed 42

python "$SCRIPT" --model autogluon --seed 42 --use_manual_ag_settings
