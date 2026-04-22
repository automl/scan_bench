#!/bin/sh

SCRIPT="plotting.py"

RESULTS_VLM="../../vlm/performance_surrogate/results/predictors"
PLOTS_VLM="../../vlm/performance_surrogate/plots"

RESULTS_TABPFN="../../tabpfn/performance_surrogate/results/predictors"
PLOTS_TABPFN="../../tabpfn/performance_surrogate/plots"

echo "Running plotting for VLM benchmark predictors comparison"
python "$SCRIPT" \
  --results-root "$RESULTS_VLM" \
  --plot-root "$PLOTS_VLM"

echo "Running plotting for TabPFN benchmark predictors comparison"
python "$SCRIPT" \
  --results-root "$RESULTS_TABPFN" \
  --plot-root "$PLOTS_TABPFN" \
  --targets "val_val_loss"

echo "Done."
