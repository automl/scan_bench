# How to Add a New Benchmark

This document describes the steps required to add a new benchmark to ScAn-Bench, under the assumption that the configuration-performance dataset has already been collected.

The setup is divided into three parts:

1. Components required for both surrogate training and API integration
2. Surrogate comparison and training
3. API integration

## 1. Setup Required for Surrogate Training and API Integration

### Step 1: Create a New Benchmark Folder

Create a new folder at the same level as the existing `vlm/` and `llm/` folders.

Name the folder according to the new model family or modality, for example:

```text
scan_benchmark/
├── vlm/
├── llm/
└── tabpfn/
````

### Step 2: Add the Collected Data

Add a `data.csv` file containing the collected configuration-performance data for the new benchmark.

For example:

```text
tabpfn/
└── data.csv
```

### Step 3: Create the Performance Surrogate Folder

Inside the newly created benchmark folder, create an empty folder named:

```text
performance_surrogate/
```

The structure should now look like:

```text
tabpfn/
├── data.csv
└── performance_surrogate/
```

### Step 4: Implement the Dataset Class

Inside `performance_surrogate/`, add a `data.py` file.

Implement a dataset class for the new model family that extends [`BaseSurrogateDataset`](scan_benchmark/dataset.py).

This class should define how the collected benchmark data is loaded and prepared for surrogate modelling.

See an example [`here`](scan_benchmark/vlm/performance_surrogate/data.py).

---

## 2. Surrogate Comparison and Training

These steps are required if you want to train and compare different surrogate models to determine which surrogate is best suited for the new benchmark.

### Step 5: Define the Data Splitter

Implement a data splitter that defines the training and test splits used for surrogate evaluation.

The structure should now look like:

```text
tabpfn/
├── data.csv
├── performance_surrogate/
└── data_splitter.py
```

The splitting strategy should reflect the characteristics of the benchmark and provide meaningful train/test partitions for comparing different surrogate models.

See an example [`here`](scan_benchmark/vlm/data_splitter.py).

### Step 6: Add the Surrogate Training Entry Point

Create a `train/` folder inside `performance_surrogate/` and add a `train.py` file.

The structure should look like:

```text
performance_surrogate/
├── data.py
└── train/
    └── train.py
```

The `train.py` script should call the generic `run_benchmark()` method.

This method handles the training and comparison of the supported surrogate models.

See an example [`here`](scan_benchmark/vlm/performance_surrogate/train/train.py).

---

## 3. API Integration

The following steps are required to expose the new benchmark through the ScAn-Bench API.

### Step 7: Define the Search Space

Add a `spaces.py` file that defines the search space of the new benchmark.

This should specify the parameters and valid ranges or choices that make up a benchmark configuration.

See an example [`here`](scan_bench/vlm/spaces.py).

### Step 8: Define the Configuration Class

Add a `config.py` file containing a configuration class that extends `BaseConfig`.

The configuration class defines the inputs required from the user to construct a valid configuration that can be passed to the surrogate model.

See an example [`here`](scan_bench/vlm/config.py).

### Step 9: Define the Available Targets

Define an `Enum` containing the prediction targets supported by the benchmark.

This can, for example, be defined inside `config.py` alongside the configuration class.

See an example [`here`](scan_bench/vlm/spaces.py).

### Step 10: Implement the Benchmark API

Add an `api.py` file and implement a benchmark class that extends `BasePerformanceBenchmark`.

By extending `BasePerformanceBenchmark`, the new benchmark can reuse the existing surrogate modelling and feature-mapping functionality.

See an example [`here`](scan_bench/vlm/api.py).

---

## Expected Structure

After completing the steps above, the new benchmark should have a structure similar to:

```text
tabpfn/
├── data.csv
├── spaces.py
├── config.py
├── api.py
└── performance_surrogate/
    ├── data.py
    └── train/
        └── train.py
```

The exact structure may vary depending on benchmark-specific requirements. The existing `vlm` implementation can be used as a reference implementation.
