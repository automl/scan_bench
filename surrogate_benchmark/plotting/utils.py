import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_matrix(corr_mat, keys, filename):
    short_keys = [shorten_label(k) for k in keys]
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
