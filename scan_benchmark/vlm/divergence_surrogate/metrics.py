import os

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, \
    average_precision_score, precision_recall_curve, auc, roc_curve


def classification_metrics(y_test, y_pred, y_prob, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=["0", "1"])

    tn, fp, fn, tp = cm.ravel()

    cm_file = os.path.join(output_dir, "confusion_matrix.png")
    plt.figure(figsize=(8, 6))
    cmd.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.savefig(cm_file)
    plt.close()

    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"TN: {tn}\n")
        f.write(f"FP: {fp}\n")
        f.write(f"FN: {fn}\n")
        f.write(f"TP: {tp}\n")

    report = classification_report(y_test, y_pred, digits=4)

    with open(metrics_file, "a") as f:
        f.write("\nClassification report:\n")
        f.write(report)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    with open(metrics_file, "a") as f:
        f.write(f"\nROC AUC: {roc_auc}\n")
        f.write(f"PR AUC: {pr_auc}\n")

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc_val = auc(recall, precision)

    pr_curve_file = os.path.join(output_dir, "precision_recall_curve.png")
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc_val:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(pr_curve_file)
    plt.close()

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    roc_curve_file = os.path.join(output_dir, "roc_curve.png")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(roc_curve_file)
    plt.close()
