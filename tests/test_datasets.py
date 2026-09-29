import numpy as np
import pandas as pd
import pytest

from scan_benchmark.llm.data import LLMSurrogateDataset
from scan_benchmark.vlm.divergence_surrogate.data import DivergenceDataset
from scan_benchmark.vlm.performance_surrogate.data import VLMSurrogateDataset


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def llm_row(config_id, valid_loss):
    row = {feature: 1.0 for feature in LLMSurrogateDataset.DEFAULT_FEATURES}
    row.update({"config_id": config_id, "valid_loss": valid_loss, "test_loss": valid_loss + 1})
    return row


def vlm_row(config_id, epoch, total_epochs, epoch_diverged, val_loss, train_num_samples=1_000_000):
    row = {feature: 1.0 for feature in VLMSurrogateDataset.DEFAULT_FEATURES}
    row.update(
        {
            "config_id": config_id,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "epoch_diverged": epoch_diverged,
            "val_loss": val_loss,
            "test_loss": val_loss + 1,
            "lr": 1e-4,
            "wd": 1e-2,
            "eps": 1e-8,
            "total_samples_planned": train_num_samples,
        }
    )
    return row


def test_llm_dataset_loads_train_test_and_all_data(repo_tmp_path):
    train_path = repo_tmp_path / "llm_train.csv"
    test_path = repo_tmp_path / "llm_test.csv"
    write_csv(train_path, [llm_row("a", 0.1), llm_row("b", 0.2)])
    write_csv(test_path, [llm_row("c", 0.3)])

    dataset = LLMSurrogateDataset(
        str(train_path),
        str(test_path),
        targets=["valid_loss", "test_loss"],
        apply_log_transform=False,
    )

    X_train, y_train = dataset.get_train_data()
    X_all, y_all = dataset.get_all_data()

    assert X_train.shape == (2, len(LLMSurrogateDataset.DEFAULT_FEATURES))
    assert y_train.tolist() == [[0.1, 1.1], [0.2, 1.2]]
    assert X_all.shape[0] == 3
    assert y_all.shape == (3, 2)


def test_llm_dataset_train_subset_selects_config_ids(repo_tmp_path):
    train_path = repo_tmp_path / "llm_train.csv"
    test_path = repo_tmp_path / "llm_test.csv"
    write_csv(train_path, [llm_row("a", 0.1), llm_row("a", 0.2), llm_row("b", 0.3)])
    write_csv(test_path, [llm_row("c", 0.4)])

    dataset = LLMSurrogateDataset(str(train_path), str(test_path), apply_log_transform=False)

    subset = dataset.get_train_subset_df(1)

    assert subset["config_id"].nunique() == 1
    assert set(subset["config_id"]).issubset({"a", "b"})


def test_vlm_dataset_filters_intermediate_points_for_test_split(repo_tmp_path):
    train_path = repo_tmp_path / "vlm_train.csv"
    test_path = repo_tmp_path / "vlm_test.csv"
    write_csv(train_path, [vlm_row("a", 1, 3, False, 0.1), vlm_row("a", 3, 3, False, 0.2)])
    write_csv(
        test_path,
        [
            vlm_row("b", 1, 3, False, 0.3),
            vlm_row("b", 2, 3, True, 0.4),
            vlm_row("b", 3, 3, False, 0.5),
        ],
    )

    dataset = VLMSurrogateDataset(str(train_path), str(test_path), apply_log_transform=False)

    assert dataset.train_df["epoch"].tolist() == [1, 3]
    assert dataset.test_df["epoch"].tolist() == [2, 3]


def test_vlm_dataset_can_keep_intermediate_test_points(repo_tmp_path):
    train_path = repo_tmp_path / "vlm_train.csv"
    test_path = repo_tmp_path / "vlm_test.csv"
    write_csv(train_path, [vlm_row("a", 1, 3, False, 0.1)])
    write_csv(test_path, [vlm_row("b", 1, 3, False, 0.3), vlm_row("b", 3, 3, False, 0.5)])

    dataset = VLMSurrogateDataset(
        str(train_path),
        str(test_path),
        eval_on_intermediate_points=True,
        apply_log_transform=False,
    )

    assert dataset.test_df["epoch"].tolist() == [1, 3]


def test_vlm_dataset_requires_filter_columns_when_dropping_intermediate_points(repo_tmp_path):
    train_path = repo_tmp_path / "vlm_train.csv"
    test_path = repo_tmp_path / "vlm_test.csv"
    row = vlm_row("a", 1, 3, False, 0.1)
    row.pop("epoch_diverged")
    write_csv(train_path, [vlm_row("a", 1, 3, False, 0.1)])
    write_csv(test_path, [row])

    with pytest.raises(ValueError, match="epoch_diverged"):
        VLMSurrogateDataset(str(train_path), str(test_path), apply_log_transform=False)


def test_vlm_dataset_applies_log_transform_to_default_log_columns(repo_tmp_path):
    train_path = repo_tmp_path / "vlm_train.csv"
    test_path = repo_tmp_path / "vlm_test.csv"
    write_csv(train_path, [vlm_row("a", 3, 3, False, 0.1)])
    write_csv(test_path, [vlm_row("b", 3, 3, False, 0.2)])

    dataset = VLMSurrogateDataset(str(train_path), str(test_path), apply_log_transform=True)

    assert dataset.train_df.loc[0, "lr"] == pytest.approx(np.log(1e-4))
    assert dataset.train_df.loc[0, "wd"] == pytest.approx(np.log(1e-2))
    assert dataset.train_df.loc[0, "eps"] == pytest.approx(np.log(1e-8))
    assert dataset.train_df.loc[0, "total_samples_planned"] == pytest.approx(np.log(1_000_000))


def test_divergence_dataset_loads_log_transformed_features(repo_tmp_path):
    train_path = repo_tmp_path / "div_train.csv"
    test_path = repo_tmp_path / "div_test.csv"
    row = {
        "config_id": "a",
        "lr": 1e-4,
        "wd": 1e-2,
        "beta1": 0.9,
        "beta2": 0.98,
        "eps": 1e-8,
        "warmup_fraction": 0.05,
        "vision_width": 256,
        "text_width": 256,
        "global_batch_size": 4096,
        "total_samples_planned": 1_000_000,
        "failed": 0,
    }
    write_csv(train_path, [row])
    write_csv(test_path, [{**row, "config_id": "b", "failed": 1}])

    dataset = DivergenceDataset(str(train_path), str(test_path))
    X_all, y_all = dataset.get_all_data()

    assert X_all.shape == (2, len(DivergenceDataset.DEFAULT_FEATURES))
    assert y_all.tolist() == [0, 1]
    assert dataset.train_df.loc[0, "lr"] == pytest.approx(np.log(1e-4))
