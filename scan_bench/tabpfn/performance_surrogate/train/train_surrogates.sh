#!/bin/bash

#SBATCH --job-name=train_surrogates
#SBATCH --output=scan_bench/tabpfn/performance_surrogate/train/train_surrogates_%A_%a.out
#SBATCH --error=scan_bench/tabpfn/performance_surrogate/train/train_surrogates_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --partition=testdlc2_gpu-h200
#SBATCH --cpus-per-task=8

# One array task per (model, fold). Submit from the repo root; TabPFN needs a GPU,
# the tree ensembles run on CPU:
#
#   MODELS="tabpfn" sbatch --array=0-4 --partition=mldlc2_gpu-l40s --gres=gpu:1 \
#       scan_bench/tabpfn/performance_surrogate/train/train_surrogates.sh
#   MODELS="ensemble:xgb ensemble:lightgbm ensemble:mix" sbatch --array=0-14 \
#       scan_bench/tabpfn/performance_surrogate/train/train_surrogates.sh
#
# Without SLURM_ARRAY_TASK_ID, all (model, fold) runs are executed sequentially.

set -e

SCRIPT="scan_bench.tabpfn.performance_surrogate.train.train"

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter '$PYTHON' not found. Run from the repo root after 'uv sync', or set PYTHON=..." >&2
    exit 1
fi
source .venv/bin/activate
SEED=42
read -r -a MODELS <<< "${MODELS:-tabpfn ensemble:xgb ensemble:lightgbm ensemble:mix}"
FOLDS=(1 2 3 4 5)

if [[ -z "${DEVICE:-}" ]]; then
    if command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

FIT_MODE="fit_with_intermediate"
PRED_MODE="pred_with_intermediate"
INTERMEDIATE_FLAGS=(--include_intermediate_points --eval_on_intermediate_points)

SPLITS_DIR="scan_bench/tabpfn/performance_surrogate/splits"
RESULTS_DIR="scan_bench/tabpfn/performance_surrogate/results"
MODEL_FAMILY="tabpfn"

RUNS=()
for MODEL_SPEC in "${MODELS[@]}"; do
    for FOLD_ID in "${FOLDS[@]}"; do
        RUNS+=("${MODEL_SPEC}:${FOLD_ID}")
    done
done

run_one() {
    local MODEL MODEL_VARIANT FOLD_ID MODEL_OUT_DIR
    IFS=":" read -r -a PARTS <<< "$1"

    MODEL="${PARTS[0]}"
    if [[ "$MODEL" == "ensemble" ]]; then
        MODEL_VARIANT="${PARTS[1]}"
        FOLD_ID="${PARTS[2]}"
        MODEL_OUT_DIR="${RESULTS_DIR}/ensemble/${MODEL_VARIANT}"
    else
        FOLD_ID="${PARTS[1]}"
        MODEL_OUT_DIR="${RESULTS_DIR}/${MODEL}"
    fi

    if [[ "$MODEL" == "tabpfn" && "$DEVICE" == "cpu" && -n "${SLURM_JOB_ID:-}" ]]; then
        echo "TabPFN needs a GPU. Submit it separately, e.g.:" >&2
        echo "  MODELS=\"tabpfn\" sbatch --array=0-4 --partition=mldlc2_gpu-l40s --gres=gpu:1 scan_bench/tabpfn/performance_surrogate/train/train_surrogates.sh" >&2
        exit 1
    fi

    OUT_DIR="${MODEL_OUT_DIR}/seed=${SEED}/${FIT_MODE}/${PRED_MODE}/fold_${FOLD_ID}"

    CMD=(
        "$PYTHON" -m "$SCRIPT"
        --model "$MODEL"
        --seed "$SEED"
        --device "$DEVICE"
        --out_dir "$OUT_DIR"
        --train_csv "${SPLITS_DIR}/train_fold_${FOLD_ID}.csv"
        --test_csv "${SPLITS_DIR}/test_fold_${FOLD_ID}.csv"
        --model_family "${MODEL_FAMILY}"
        "${INTERMEDIATE_FLAGS[@]}"
    )

    if [[ "$MODEL" == "ensemble" ]]; then
        CMD+=(--ensemble_type "$MODEL_VARIANT")
    fi

    echo "Running: ${CMD[*]}"
    "${CMD[@]}"
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if (( SLURM_ARRAY_TASK_ID >= ${#RUNS[@]} )); then
        echo "Array index ${SLURM_ARRAY_TASK_ID} out of range, only ${#RUNS[@]} runs: ${RUNS[*]}" >&2
        exit 1
    fi
    run_one "${RUNS[$SLURM_ARRAY_TASK_ID]}"
else
    for RUN in "${RUNS[@]}"; do
        run_one "$RUN"
    done
fi
