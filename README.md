# ScAn-Bench: Evaluating Scaling Analysis Methodology

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

We recommend using a conda environment.

```bash
conda create -n scan-benchmark python=3.11
conda activate scan-benchmark
pip install -e .[dev]
```

## Training and evaluation

To train and get the performance results for the surrogate benchmarks, run the provided shell scripts.

### VLM pipeline

```bash
bash scan_benchmark/vlm/performance_surrogate/train/train_surrogates.sh
```

### LLM pipeline

```bash
bash scan_benchmark/llm/train_surrogates.sh
```

## Results

### Surrogate Performance (VLM)

| Surrogate | RMSE ↓ | MAE ↓ | MDAE ↓ | MARPD ↓ | R² ↑ | R ↑ | Corr. ↑ |
|----------|--------|--------|--------|---------|------|------|---------|
| **TabPFN** | **0.117** | **0.061** | **0.029** | **2.579** | **0.986** | **0.993** | **0.994** |
| AutoGluon | 0.153 | 0.091 | 0.050 | 4.072 | 0.975 | 0.988 | 0.990 |
| XGB       | 0.285 | 0.201 | 0.153 | 8.302 | 0.915 | 0.958 | 0.959 |
| Mix       | 0.289 | 0.213 | 0.172 | 8.981 | 0.912 | 0.960 | 0.960 |
| LGB       | 0.337 | 0.263 | 0.224 | 11.286 | 0.880 | 0.959 | 0.959 |

### Surrogate Performance (LLM)

| Surrogate | RMSE ↓ | MAE ↓ | MDAE ↓ | MARPD ↓ | R² ↑ | R ↑ | Corr. ↑ |
|----------|--------|--------|--------|---------|------|------|---------|
| **TabPFN** | **0.098** | **0.034** | **0.008** | **1.371** | **0.958** | **0.979** | **0.995** |
| AutoGluon | 0.159 | 0.052 | 0.015 | 2.184 | 0.889 | 0.945 | 0.982 |
| XGB       | 0.196 | 0.086 | 0.036 | 3.780 | 0.831 | 0.913 | 0.964 |
| Mix       | 0.199 | 0.107 | 0.063 | 4.776 | 0.826 | 0.919 | 0.967 |
| LGB       | 0.207 | 0.122 | 0.080 | 5.591 | 0.811 | 0.917 | 0.958 |

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
