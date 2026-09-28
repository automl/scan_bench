import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_matrix(corr_mat, keys, filename):
    short_keys = [clean_task_label(k) for k in keys]
    sns.clustermap(
        corr_mat,
        xticklabels=short_keys,
        yticklabels=short_keys,
        annot=False,
        # fmt=".2f",
        # annot_kws={"size": 6},
    )

    plt.savefig(filename)
    plt.close()


TASK_NAME_MAP = {
    "vtab_caltech101_mean_per_class_recall": "Caltech-101",
    "cifar10_mean_per_class_recall": "CIFAR-10",
    "vtab_cifar100_mean_per_class_recall": "CIFAR-100",
    "vtab_clevr_count_all_acc1": "CLEVR Counts",
    "vtab_clevr_closest_object_distance_acc1": "CLEVR Distance",
    "country211_acc1": "Country211",
    "vtab_dtd_acc1": "Describable Textures",
    "vtab_eurosat_acc1": "EuroSAT",
    "fgvc_aircraft_mean_per_class_recall": "FGVC Aircraft",
    "food101_acc1": "Food-101",
    "gtsrb_acc1": "GTSRB",
    "imagenet1k_acc1": "ImageNet 1k",
    "imagenet_sketch_acc1": "ImageNet Sketch",
    "imagenetv2_acc1": "ImageNet v2",
    "imagenet-a_acc1": "ImageNet-A",
    "imagenet-o_acc1": "ImageNet-O",
    "imagenet-r_acc1": "ImageNet-R",
    "vtab_kitti_closest_vehicle_distance_acc1": "KITTI Vehicle Distance",
    "mnist_acc1": "MNIST",
    "objectnet_acc1": "ObjectNet",
    "vtab_flowers_mean_per_class_recall": "Oxford Flowers-102",
    "vtab_pets_mean_per_class_recall": "Oxford-IIIT Pet",
    "voc2007_acc1": "Pascal VOC 2007",
    "vtab_pcam_acc1": "PatchCamelyon",
    "renderedsst2_acc1": "Rendered SST2",
    "vtab_resisc45_acc1": "RESISC45",
    "cars_acc1": "Stanford Cars",
    "stl10_acc1": "STL-10",
    "sun397_acc1": "SUN397",
    "vtab_svhn_acc1": "SVHN",
    "retrieval_flickr_1k_test_image_text_retrieval_mean_recall@1":
        "Flickr",
    "retrieval_mscoco_2014_5k_test_image_text_retrieval_mean_recall@1":
        "MSCOCO",
    "misc_winogavil_jaccard_score_10-12": "WinoGAViL",
    "wilds_iwildcam_F1-macro_all": "iWildCam",
    "wilds_camelyon17_acc1": "Camelyon17",
    "wilds_fmow_acc_worst_region": "FMoW",
    "fairness_dollar_street_acc_top5_wg": "Dollar Street",
    "fairness_geode_acc_wg": "GeoDE",
    "fairness_fairface_acc_race_avg": "FairFace",
    "fairness_utkface_acc_race_avg": "UTKFace",
    "val_loss": "Validation loss",
    "test_loss": "Test loss",
}


def clean_task_label(task: str) -> str:
    return TASK_NAME_MAP.get(task, task)


def shorten_label(label):
    return label.replace("retrieval_", "") \
        .replace("vtab_", "") \
        .replace("_mean_per_class_recall", "") \
        .replace("_mean_recall@1", "") \
        .replace("_acc1", "") \
        .replace("_acc_wg", "_wg") \
        .replace("_acc_top5_wg", "") \
        .replace("_acc_worst_region", "") \
        .replace("_F1-macro_all", "") \
        .replace("_acc_race_avg", "") \
        .replace("_jaccard_score_10-12", "") \
        .replace("mscoco_2014_5k_test_image_text", "mscoco") \
        .replace("flickr_1k_test_image_text", "flickr") \
        .replace("kitti_closest_vehicle_distance", "kitti_cvd") \
        .replace("clevr_closest_object_distance", "clevr_cod")
