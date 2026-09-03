import mlflow
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

def log_evaluation_figures(
    y_true,
    y_probas,
    bac_values,
    pr_precision,
    pr_recall,
    confusion_matrix: tuple[int, int, int, int],
    auroc: float,
    has_both_classes: bool,
    eval_type: str,
    epoch: int,
) -> None:

    # Confusion Matrix
    tp, tn, fp, fn = confusion_matrix
    matrix = np.array([[tn, fp], [fn, tp]])
    cm_fig, cm_ax = plt.subplots(figsize=(5, 4), dpi=120)
    image = cm_ax.imshow(matrix, cmap="Blues")
    cm_fig.colorbar(image, ax=cm_ax)
    cm_ax.set(
        title=f"{eval_type} confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Sober", "Intoxicated"],
        yticklabels=["Sober", "Intoxicated"],
    )
    for row, column in np.ndindex(matrix.shape):
        cm_ax.text(
            column,
            row,
            matrix[row, column],
            ha="center",
            va="center",
            color="white" if matrix[row, column] > matrix.max() / 2 else "black",
        )
    _log_figure_with_step(cm_fig, f"{eval_type}_confusion_matrix", epoch)

    # Precision-Recall Curve
    pr_auc = auc(pr_recall, pr_precision)
    pr_fig, pr_ax = plt.subplots(figsize=(5, 4), dpi=120)
    pr_ax.plot(pr_recall, pr_precision, label=f"AUC={pr_auc:.4f}")
    pr_ax.set_title(f"{eval_type} precision-recall")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_xlim(0.0, 1.0)
    pr_ax.set_ylim(0.0, 1.05)
    pr_ax.legend(loc="lower left")
    pr_ax.grid(alpha=0.3)
    _log_figure_with_step(pr_fig, f"{eval_type}_precision_recall_curve", epoch)

    # Receiver Operating Characteristic (ROC) curve
    roc_fig, roc_ax = plt.subplots(figsize=(5, 4), dpi=120)
    if has_both_classes:
        fpr, tpr, _ = roc_curve(y_true, y_probas)
        roc_ax.plot(fpr, tpr, label=f"AUROC={auroc:.4f}")
    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    roc_ax.set_title(f"{eval_type} ROC")
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_xlim(0.0, 1.0)
    roc_ax.set_ylim(0.0, 1.05)
    roc_ax.legend(loc="lower right")
    roc_ax.grid(alpha=0.3)
    _log_figure_with_step(roc_fig, f"{eval_type}_roc_curve", epoch)

    # BAC versus predicted intoxication probability
    intoxicated_mask = bac_values != 0
    bac_fig, bac_ax = plt.subplots(figsize=(5, 4), dpi=120)
    if intoxicated_mask.any():
        bac_ax.scatter(
            bac_values[intoxicated_mask],
            y_probas[intoxicated_mask],
            alpha=0.65,
        )
    else:
        bac_ax.text(
            0.5,
            0.5,
            "No intoxicated samples",
            ha="center",
            va="center",
            transform=bac_ax.transAxes,
        )
    bac_ax.set(
        title=f"{eval_type} BAC versus P(drunk)",
        xlabel="BAC (‰)",
        ylabel="P(drunk)",
        ylim=(-0.02, 1.02),
    )
    bac_ax.grid(alpha=0.3)
    _log_figure_with_step(bac_fig, f"{eval_type}_bac_probability", epoch)


def _log_figure_with_step(fig, image_key: str, epoch: int) -> None:
    fig.tight_layout()
    suffix = f"epoch_{epoch}" if epoch is not None else "final"
    mlflow.log_figure(fig, f"images/{image_key}_{suffix}.png")
    plt.close(fig)
