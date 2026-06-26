#!/bin/bash

SCRIPT="scan_benchmark.vlm.performance_surrogate.train.train"

SEEDS=(42)
ENSEMBLE_TYPES=("xgb" "lightgbm" "mix")
DEVICE="cuda"

SPLITS_DIR="scan_benchmark/vlm/performance_surrogate/splits"
RESULTS_DIR="scan_benchmark/vlm/performance_surrogate/results"
ADDITIONAL_RUNS_PATH="scan_benchmark/vlm/large_runs.csv"

EXPERIMENTS=(
#    "fit_no_intermediate pred_no_intermediate"
#    "fit_with_intermediate pred_no_intermediate"
    "fit_with_intermediate pred_with_intermediate"
)

FOLDS=(1 2 3 4 5)

for SEED in "${SEEDS[@]}"; do

    for MODEL in tabpfn ensemble; do

        if [[ "$MODEL" == "tabpfn" ]]; then
            MODEL_VARIANTS=("tabpfn")
        else
            MODEL_VARIANTS=("${ENSEMBLE_TYPES[@]}")
        fi

        for MODEL_VARIANT in "${MODEL_VARIANTS[@]}"; do

            if [[ "$MODEL" == "tabpfn" ]]; then
                MODEL_OUT_DIR="${RESULTS_DIR}/tabpfn"
            else
                MODEL_OUT_DIR="${RESULTS_DIR}/ensemble/${MODEL_VARIANT}"
            fi

            for EXPERIMENT in "${EXPERIMENTS[@]}"; do
                read -r FIT_MODE PRED_MODE <<< "$EXPERIMENT"

                for FOLD_ID in "${FOLDS[@]}"; do

                    OUT_DIR="${MODEL_OUT_DIR}/seed=${SEED}/${FIT_MODE}/${PRED_MODE}/fold_${FOLD_ID}"

                    CMD=(
                        python -m "$SCRIPT"
                        --model "$MODEL"
                        --seed "$SEED"
                        --labels val_loss
                        --device "$DEVICE"
                        --out_dir "$OUT_DIR"
                        --train_csv "${SPLITS_DIR}/train_fold_${FOLD_ID}.csv"
                        --test_csv "${SPLITS_DIR}/test_fold_${FOLD_ID}.csv"
                        --additional_runs_path "${ADDITIONAL_RUNS_PATH}"
                    )

                    if [[ "$MODEL" == "ensemble" ]]; then
                        CMD+=(--ensemble_type "$MODEL_VARIANT")
                    fi

                    if [[ "$FIT_MODE" == "fit_with_intermediate" ]]; then
                        CMD+=(--include_intermediate_points)
                    fi

                    if [[ "$PRED_MODE" == "pred_with_intermediate" ]]; then
                        CMD+=(--eval_on_intermediate_points)
                    fi

                    echo "Running: ${CMD[*]}"
                    "${CMD[@]}"

                done
            done
        done
    done
done

for SEED in "${SEEDS[@]}"; do

    MODEL="autogluon"

    MODEL_OUT_DIR="${RESULTS_DIR}/autogluon"

    FIT_MODE="fit_with_intermediate"
    PRED_MODE="pred_with_intermediate"

    for FOLD_ID in "${FOLDS[@]}"; do

        OUT_DIR="${MODEL_OUT_DIR}/seed=${SEED}/${FIT_MODE}/${PRED_MODE}/fold_${FOLD_ID}"

        CMD=(
            python -m "$SCRIPT"
            --model "$MODEL"
            --seed "$SEED"
            --labels val_loss
            --device "$DEVICE"
            --out_dir "$OUT_DIR"
            --train_csv "${SPLITS_DIR}/train_fold_${FOLD_ID}.csv"
            --test_csv "${SPLITS_DIR}/test_fold_${FOLD_ID}.csv"
            --include_intermediate_points
            --eval_on_intermediate_points
            --additional_runs_path "${ADDITIONAL_RUNS_PATH}"
        )

        echo "Running: ${CMD[*]}"
        "${CMD[@]}"

    done
done
