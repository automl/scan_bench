import argparse

import pandas as pd

from scan_benchmark.commons.metrics.metrics import calculate_correlation_matrix
from scan_benchmark.commons.plotting.utils import plot_correlation_matrix

METRIC_MAP = {
    "vtab_caltech101": "mean_per_class_recall",
    "cifar10": "mean_per_class_recall",
    "vtab_cifar100": "mean_per_class_recall",
    "vtab_clevr_count_all": "acc1",
    "vtab_clevr_closest_object_distance": "acc1",
    "country211": "acc1",
    "vtab_dtd": "acc1",
    "vtab_eurosat": "acc1",
    "fgvc_aircraft": "mean_per_class_recall",
    "food101": "acc1",
    "gtsrb": "acc1",
    "imagenet1k": "acc1",
    "imagenet_sketch": "acc1",
    # "imagenetv2": "acc1",
    "imagenet-a": "acc1",
    "imagenet-o": "acc1",
    "imagenet-r": "acc1",
    "vtab_kitti_closest_vehicle_distance": "acc1",
    "mnist": "acc1",
    "objectnet": "acc1",
    "vtab_flowers": "mean_per_class_recall",
    "vtab_pets": "mean_per_class_recall",
    "voc2007": "acc1",
    "vtab_pcam": "acc1",
    "renderedsst2": "acc1",
    "vtab_resisc45": "acc1",
    "cars": "acc1",
    "stl10": "acc1",
    "sun397": "acc1",
    "vtab_svhn": "acc1",
    "retrieval_flickr_1k_test_image_text_retrieval": "mean_recall@1",
    "retrieval_mscoco_2014_5k_test_image_text_retrieval": "mean_recall@1",
    "misc_winogavil": "jaccard_score_10-12",
    "wilds_iwildcam": "F1-macro_all",
    "wilds_camelyon17": "acc1",
    "wilds_fmow": "acc_worst_region",
    "fairness_dollar_street": "acc_top5_wg",
    "fairness_geode": "acc_wg",
    "fairness_fairface": "acc_race_avg",
    "fairness_utkface": "acc_race_avg",
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", default="corr_matrix.png")

    args = parser.parse_args()

    train_df = pd.read_csv("../../vlm/performance_surrogate/splits/train_fold_1.csv")
    test_df = pd.read_csv("../../vlm/performance_surrogate/splits/test_fold_1.csv")

    df = pd.concat([train_df, test_df], ignore_index=True)

    df["config_id"] = pd.to_numeric(df["config_id"], errors="ignore")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="ignore")

    df = df.sort_values(["config_id", "epoch"]).reset_index(drop=True)

    downstream_cols = []
    for dataset, metric in METRIC_MAP.items():
        col_name = f"{dataset}_{metric}"
        if col_name in df.columns:
            downstream_cols.append(col_name)

    base_cols = ["val_loss", "test_loss"]
    cols = base_cols + downstream_cols
    df = df[cols]

    for col in downstream_cols:
        df[col] = 1 - df[col]

    df = df.dropna()
    data_dict = {col: df[col].tolist() for col in cols}
    corr_mat, keys = calculate_correlation_matrix(data_dict)
    plot_correlation_matrix(corr_mat, keys, args.out)


if __name__ == "__main__":
    main()
