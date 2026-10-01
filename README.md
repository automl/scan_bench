# ScAn-Bench: Surrogate Benchmarks for Scaling Analysis

[![PyPI version](https://img.shields.io/pypi/v/scan-bench?color=informational)](https://pypi.org/project/scan-bench/)
[![Python versions](https://img.shields.io/pypi/pyversions/scan-bench)](https://pypi.org/project/scan-bench/)
[![License](https://img.shields.io/pypi/l/scan-bench?color=informational)](https://github.com/automl/scan_bench/blob/main/LICENSE)
[![Tests](https://github.com/automl/scan_bench_suite/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/automl/scan_bench_suite/actions/workflows/tests.yaml)

<p align="center">
  <img src="https://raw.githubusercontent.com/automl/scan_bench/main/doc_figures/empirical_pareto_fronts.png" alt="Training runs, empirical Pareto fronts and power-law fits for the OpenCLIP, LLM and TabPFN benchmarks" width="100%">
</p>
<p align="center"><em>The three ScAn-Bench suites. Grey: training runs. Orange: empirical compute-optimal front. Dotted: power-law fit.</em></p>

**Scaling studies are expensive, and every new method pays for them again.** To validate a scaling analysis method,
groups train hundreds of models, burn thousands of GPU-hours (and the energy that comes with them), then throw the sweep
away. The next paper repeats it from scratch.

Existing benchmarks also only cover half of the problem. They either tune hyperparameters such as learning rate and
batch size for one **fixed architecture**, or scale the **architecture** while fixing those hyperparameters to
heuristic values. Scaling analysis needs both to vary together.

**ScAn-Bench does the expensive part once.** We trained **5,352 configurations** across
three model families, jointly varying model scale, data scale and training hyperparameters over 2–3 orders of
magnitude of compute. Surrogate models fitted to these runs let you query any configuration in the search space, at any
scale, in **under 20 seconds**, with no GPU required.

| Benchmark | Model family | Configurations | Compute range (FLOPs) | Targets |
|---|---|---:|---|---|
| **VLM** | OpenCLIP | 1,535 | 10<sup>14</sup> – 10<sup>16</sup> | val/test loss, divergence report |
| **LLM** | Decoder-only Transformer | 1,194 | 10<sup>16</sup> – 10<sup>19</sup> | val/test loss |
| **TabPFN** | Encoder-only Tabular foundation model | 2,623 | 10<sup>11</sup> – 10<sup>14</sup> | prior validation loss |

For a general overview of the paper, repository structure and artifact maps refer to [PaperOverview](https://github.com/automl/scan_bench/blob/main/PaperOverview.md).

To create a new benchmark based on this framework refer to [BenchmarkCreation](https://github.com/automl/scan_bench/blob/main/BenchmarkCreation.md).

## ScAn-Bench Repository


### SDK Installation

We recommend using a conda environment for installing the SDK:

```bash 
pip install scan-bench
```

Install pytorch with CUDA, if you want to utilize the GPU.

### Quick start

```python
from scan_bench import TabPFNBenchmark, TabPFNConfig, TabPFNTarget, PerformancePredictorType

bench = TabPFNBenchmark(
    target=TabPFNTarget.VAL_LOSS,
    predictor_type=PerformancePredictorType.ENSEMBLE_XGB,  # or TABPFN, ENSEMBLE_LIGHTGBM, ENSEMBLE_MIX, AUTOGLUON
    device="auto",                                         # or "cpu", "cuda"
)

small = TabPFNConfig(total_cells=2**20, effective_batch_size=16, lr=1e-4,
                     max_features=32, embedding_size=4, num_layers=2, num_datapoints_max=128)
large = TabPFNConfig(total_cells=2**30, effective_batch_size=64, lr=1e-3,
                     max_features=64, embedding_size=64, num_layers=8, num_datapoints_max=256)

print(bench.query(small))                  # {"predictions": {"mean", "uncertainty"}, "model_stats": {"flops", ...}}
print(bench.query_many([small, large]))    # batched, one result per config
print(bench.flops(large), bench.model_params(large))
```

### Package usage

Refer to [VLM API](https://github.com/automl/scan_bench/blob/main/scan_bench/vlm/api.py), [LLM API](https://github.com/automl/scan_bench/blob/main/scan_bench/llm/api.py) and [TabPFN API](https://github.com/automl/scan_bench/blob/main/scan_bench/tabpfn/api.py) for API usages.

### Local development and experiment reproduction

For local development and experiment reproduction, use the cloned repository. The training, evaluation, and plotting
shell scripts described below are source-tree workflows and should be run from the repository root.

#### Surrogate training and evaluation

To train and get the performance results for the surrogate benchmarks, run the provided shell scripts. Change DEVICE
to 'cuda' in train_surrogates.sh to use the GPU.

#### VLM pipeline

##### VLM performance predictor surrogate

```bash
bash scan_bench/vlm/performance_surrogate/train/train_surrogates.sh
```

##### VLM divergences predictor surrogate

```bash
bash scan_bench/vlm/divergence_surrogate/train.sh
```

#### LLM pipeline

```bash
bash scan_bench/llm/train_surrogates.sh
```

#### TabPFN pipeline

```bash
bash scan_bench/tabpfn/performance_surrogate/train/train_surrogates.sh
```

#### Results

After running the surrogate training scripts, a results directory is created under the corresponding directory for each
model family. The generated subdirectories contain JSON files with the evaluation results for the trained surrogate
models. These JSON outputs include the metrics reported below and are used to construct the corresponding tables in the
paper.

For example:

scan_bench/vlm/performance_surrogate/results/

scan_bench/llm/results/

scan_bench/tabpfn/performance_surrogate/results/

#### Surrogate Performance (VLM)

| Surrogate  | RMSE ↓    | MAE ↓     | MDAE ↓    | MARPD ↓   | R² ↑      | R ↑       | Corr. ↑   |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| **TabPFN** | **0.21** | **0.06** | **0.02** | **2.69** | **0.96** | **0.98** | 0.98 |
| AutoGluon  | 0.26     | 0.12     | 0.06     | 5.79     | 0.94     | 0.97     | 0.98    |
| XGB        | 0.35     | 0.20     | 0.12     | 9.11     | 0.90     | 0.95     | 0.96     |
| Mix        | 0.37     | 0.21     | 0.14     | 10.11     | 0.88     | 0.94     | 0.95     |
| LGB        | 0.42     | 0.29     | 0.24     | 14.88    | 0.86     | 0.94     | 0.95    |

#### Surrogate Performance (LLM)

| Surrogate  | RMSE ↓    | MAE ↓     | MDAE ↓    | MARPD ↓   | R² ↑      | R ↑       | Corr. ↑   |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| **TabPFN** | **0.27** | **0.08** | **0.01** | **3.02** | **0.77** | **0.88** | **0.95** |
| AutoGluon  | 0.30     | 0.11     | 0.02     | 4.66    | 0.72     | 0.85    | 0.91     |
| XGB        | 0.35     | 0.15     | 0.04     | 6.80   | 0.63    | 0.80    | 0.86     |
| Mix        | 0.33     | 0.17     | 0.07     | 7.65   | 0.66     | 0.82     | 0.86     |
| LGB        | 0.33     | 0.17     | 0.08     | 7.97     | 0.66     | 0.83     | 0.87     |

### Data

The repository includes pre-collected configuration-performance datasets used to train the surrogate models.

#### VLM dataset summary

| Quantity| Count | Description |
|---|---:|---|
| Training configurations | 1,535 | Total number of collected VLM training configurations. |
| Failed configurations | 134| Configurations that diverged during training. |
| Successful configurations | 1366| Configurations that completed training successfully. |
| Collected checkpoint rows | 8,024 | Total checkpoint rows collected across all runs. |
| Successful checkpoint rows | 7,701 | Checkpoints from successful runs |



#### LLM dataset summary

| Quantity | Count | Description |
|---|---:|---|
| Training configurations | 1,194 | Total number of collected LLM training configurations. |
| Collected checkpoint rows | 4,524 | Total checkpoints collected across successful and failed runs. |
| Performance-surrogate rows | 4,524 | Checkpoints from successful runs used for performance prediction. |

#### TabPFN dataset summary

| Quantity | Count | Description |
|---|---:|---|
| Training configurations | 2,623 | Total number of collected TabPFN pretraining configurations (one row per configuration). |


Each configuration varies the hyperparameters (`lr`, `effective_batch_size`) and the scale parameters (`total_cells`,
`embedding_size`, `num_layers`, `max_features`, `num_datapoints_max`); see
[TabPFN search space](https://github.com/automl/scan_bench/blob/main/scan_bench/tabpfn/spaces.py). Available target is the prior validation loss (`val/val_loss`).

#### Data locations

The table below shows where the data is located:

| Dataset | Path | Description |
|---|---|---|
| **VLM performance data** | `scan_bench/vlm/performance_surrogate/splits` | Training and test splits for VLM performance surrogate modeling. |
| **VLM divergence data** | `scan_bench/vlm/divergence_surrogate/splits` | Configuration-level data for predicting failed (diverged) configurations. |
| **LLM performance data** | `scan_bench/llm/splits` | Configuration-performance datasets for LLM surrogate training. |
| **TabPFN performance data** | `scan_bench/tabpfn/performance_surrogate/splits` | Configuration-performance datasets for TabPFN surrogate training. |


Additionally, we host the datasets online, with the corresponding Licenses, source dataset Licenses and corresponding downstream task Licenses:

[VLM-Dataset](https://www.kaggle.com/datasets/4eca11cf1ddfee40dcc21b9dd6fc7f0f59ffd074f3691440c231ff68eb2a93bd).

[LLM-Dataset](https://kaggle.com/datasets/8e8a59d2a9f9c2abfe47d22e8ce78c99e24bb524b042254e8da312262d1503eb).


The Croissant RAI metadata files for each dataset are included in this repository:

[VLM-Croissant-RAI](https://github.com/automl/scan_bench/blob/main/scan_bench/vlm/croissant_rai_ScAn_VLM-Bench.json)
[LLM-Croissant-RAI](https://github.com/automl/scan_bench/blob/main/scan_bench/llm/croissant_rai_ScAn_LLM-Bench.json)

### Unit testing
To run the unit tests provided in [tests/](https://github.com/automl/scan_bench/tree/main/tests) make sure that pytest is installed.

```bash
pip install pytest
```
Run the tests with the following command:

```bash
python -m pytest
```

### Contributing

Contributions are welcome. Please open an issue or submit a pull request.

For usage and licensing terms, see the LICENSE file.


### Citations

If you use ScAn-Bench, please cite:

```bibtex
@inproceedings{sermaxhaj2026scanbench,
  title={ScAn-Bench: Evaluating Scaling Analysis Methodology},
  author={A. Sermaxhaj and N. Alipour and D. Sinani and J. Hog and N. Mallik and S. Adriaensen
          and J. Jitsev and D. Stoll},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  note={Evaluations and Datasets Track},
  url={https://arxiv.org/abs/2609.35707}
}
```
and

```bibtex
@inproceedings{alipour2026tabpfn,
  title={TabPFN-ScAn-Bench: A Surrogate Benchmark for Scaling Analysis Algorithms},
  author={N. Alipour and D. Sinani and A. Sermaxhaj and J. Hog and D. Stoll},
  booktitle={AutoML 2026 non-archival},
  year={2026},
  note={Late-Breaking Abstract}
}
```
