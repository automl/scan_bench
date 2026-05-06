#!/bin/sh

SCRIPT="scan_benchmark.commons.plotting.plotting"

RESULTS_VLM="scan_benchmark/vlm/performance_surrogate/results/predictors"
PLOTS_VLM="scan_benchmark/vlm/performance_surrogate/plots"

RESULTS_LLM="scan_benchmark/llm/results/predictors"
PLOTS_LLM="scan_benchmark/llm/plots"

echo "Running plotting for VLM benchmark predictors comparison"
python -m  "$SCRIPT" \
  --results-root "$RESULTS_VLM" \
  --plot-root "$PLOTS_VLM"

echo "Running plotting for LLM benchmark predictors comparison"
python "$SCRIPT" \
  --results-root "$RESULTS_LLM" \
  --plot-root "$PLOTS_LLM" \
  --targets "test_loss"

echo "Done."
