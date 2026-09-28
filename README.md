# ScAn-Bench: Evaluating Scaling Analysis Methodology

This repository provides surrogate benchmarks for evaluating scaling analysis (ScAn) methodology on vision-language
models (VLMs), large language models (LLMs) and tabular foundation models (TabPFN). The benchmarks approximate the mapping from training configurations to
performance, enabling fast evaluation without training full models.

This README describes the ScAn-Bench repository specifically.

For a general overview of the paper, repository structure and artifact maps refer to [PaperOverview](https://anonymous.4open.science/r/scan_bench_suite-3E3D/PaperOverview.md).

To create a new benchmark based on this framework refer to [BenchmarkCreation](https://anonymous.4open.science/r/scan_bench_suite-3E3D/BenchmarkCreation.md).

## ScAn-Bench Repository


### Installation

ScAn-Bench can be used in two ways: as an installed Python package for API querying, or from the cloned source
repository for local development and experiment reproduction.

We recommend using a conda environment for both package usage and local development:

```bash 
conda create -n scan-bench python=3.11
conda activate scan-bench
pip install scan-bench
```

Install pytorch with CUDA, if you want to utilize the GPU.

### Quick start

```python
from scan_bench import TabPFNBenchmark, TabPFNConfig, TabPFNTarget, PerformancePredictorType

bench = TabPFNBenchmark(
    target=TabPFNTarget.VAL_LOSS,                          # or NLL, ROC_AUC
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

Refer to [VLM API](scan_bench/vlm/api.py), [LLM API](scan_bench/llm/api.py) and [TabPFN API](scan_bench/tabpfn/api.py) for API usages.

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

For raw logs on the collected VLM data, see the [ScAn-VLM-Bench repository](https://anonymous.4open.science/r/scan_vlm_bench-C6CB/README.md#Additional).


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
[TabPFN search space](scan_bench/tabpfn/spaces.py). Available target is the prior validation loss (`val/val_loss`).

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

[VLM-Croissant-RAI](https://anonymous.4open.science/r/scan_bench_suite-3E3D/scan_benchmark/vlm/croissant_rai_ScAn-VLM-Bench.json)
[LLM-Croissant-RAI](https://anonymous.4open.science/r/scan_bench_suite-3E3D/scan_benchmark/llm/croissant_rai_LLM-ScAn-Bench.json)

### Additional

### Unit testing
To run the unit tests provided in [tests/](https://anonymous.4open.science/r/scan_bench_suite-3E3D/tests) make sure that pytest is installed.

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
  author={A. Sermaxhaj and N. Alipour and D. Sinani and J. Hog and N. Mallik and S. Adriaensen and J. Jitsev and D. Stoll},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  note={Evaluations and Datasets Track},
  url={https://openreview.net/forum?id=EQd9HNVF60}
}
```
and

```bibtex
@inproceedings{alipour2026tabpfn,
  title={TabPFN-ScAn-Bench: A Surrogate Benchmark for Scaling Analysis Algorithms},
  author={N. Alipour and D. Sinani and A. Sermaxhaj and J. Hog and D. Stoll},
  booktitle={AutoML 2026},
  year={2026},
  note={Late-Breaking Abstract}
}
```