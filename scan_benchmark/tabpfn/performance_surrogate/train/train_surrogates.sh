#!/bin/bash

SCRIPT="scan_benchmark.tabpfn.performance_surrogate.train.train"

SEEDS=(42)
ENSEMBLE_TYPES=("xgb" "lightgbm" "mix")
DEVICE="cuda"

FIT_MODE="fit_with_intermediate"
PRED_MODE="pred_with_intermediate"

SPLITS_DIR="scan_benchmark/tabpfn/performance_surrogate/splits"
RESULTS_DIR="scan_benchmark/tabpfn/performance_surrogate/results"
MODEL_FAMILY="tabpfn"

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

              for FOLD_ID in "${FOLDS[@]}"; do

                  OUT_DIR="${MODEL_OUT_DIR}/seed=${SEED}/${FIT_MODE}/${PRED_MODE}/fold_${FOLD_ID}"

                  CMD=(
                        python -m "$SCRIPT"
                        --model "$MODEL"
                        --seed "$SEED"
                        --device "$DEVICE"
                        --out_dir "$OUT_DIR"
                        --train_csv "${SPLITS_DIR}/train_fold_${FOLD_ID}.csv"
                        --test_csv "${SPLITS_DIR}/test_fold_${FOLD_ID}.csv"
                        --model_family "${MODEL_FAMILY}"
                    )

                  if [[ "$MODEL" == "ensemble" ]]; then
                        CMD+=(--ensemble_type "$MODEL_VARIANT")
                  fi

                  echo "Running: ${CMD[*]}"
                  "${CMD[@]}"

              done
        done
    done
done

for SEED in "${SEEDS[@]}"; do

    MODEL="autogluon"

    MODEL_OUT_DIR="${RESULTS_DIR}/autogluon"


    for FOLD_ID in "${FOLDS[@]}"; do

        OUT_DIR="${MODEL_OUT_DIR}/seed=${SEED}/${FIT_MODE}/${PRED_MODE}/fold_${FOLD_ID}"

        CMD=(
            python -m "$SCRIPT"
            --model "$MODEL"
            --seed "$SEED"
            --device "$DEVICE"
            --out_dir "$OUT_DIR"
            --train_csv "${SPLITS_DIR}/train_fold_${FOLD_ID}.csv"
            --test_csv "${SPLITS_DIR}/test_fold_${FOLD_ID}.csv"
            --model_family "${MODEL_FAMILY}"
        )

        echo "Running: ${CMD[*]}"
        "${CMD[@]}"

    done
done
