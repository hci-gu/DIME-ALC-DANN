import mlflow
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from matplotlib.ticker import MaxNLocator

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
    audio_lengths: np.ndarray | None = None,
    speaker_ids: np.ndarray | None = None,
    threshold: float = 0.5,
    bac_limit: float | None = None,
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
        max_bac = float(bac_values[intoxicated_mask].max())
        bac_ax.scatter(
            bac_values[intoxicated_mask],
            y_probas[intoxicated_mask],
            alpha=0.65,
        )
    else:
        max_bac = 1.0
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
        xlim=(0.0, 1.1 * max_bac),
        ylim=(-0.02, 1.02),
    )
    bac_ax.grid(alpha=0.3)
    _log_figure_with_step(bac_fig, f"{eval_type}_bac_probability", epoch)

    # Audio duration versus predicted intoxication probability (all samples)
    if audio_lengths is not None:
        length_fig, length_ax = plt.subplots(figsize=(5, 4), dpi=120)
        length_ax.scatter(audio_lengths, y_probas, alpha=0.65)
        length_ax.set(
            title=f"{eval_type} audio length versus P(intoxicated)",
            xlabel="Audio length (s)",
            ylabel="P(intoxicated)",
            ylim=(-0.02, 1.02),
        )
        length_ax.set_xlim(left=0.0)
        length_ax.grid(alpha=0.3)
        _log_figure_with_step(
            length_fig, f"{eval_type}_audio_length_probability", epoch
        )

    # Balanced accuracy for each speaker using the evaluation threshold.
    if speaker_ids is not None:
        speakers = np.unique(speaker_ids)
        y_pred = y_probas >= threshold
        scores = []
        for speaker in speakers:
            mask = speaker_ids == speaker
            labels = y_true[mask].astype(bool)
            predictions = y_pred[mask]
            if not labels.any() or labels.all():
                scores.append(np.nan)
                continue
            sensitivity = predictions[labels].mean()
            specificity = (~predictions[~labels]).mean()
            scores.append((sensitivity + specificity) / 2)

        speaker_fig, speaker_ax = plt.subplots(
            figsize=(max(6, 0.35 * len(speakers)), 4), dpi=120
        )
        positions = np.arange(len(speakers))
        speaker_ax.bar(positions, scores)
        for position, score in zip(positions, scores):
            if np.isnan(score):
                speaker_ax.text(
                    position, 0.02, "N/A", ha="center", va="bottom",
                    rotation=90, fontsize=8,
                )
        speaker_ax.set(
            title=f"{eval_type} balanced accuracy by speaker",
            xlabel="Speaker ID (N/A: only one class present)",
            ylabel="Balanced accuracy",
            xticks=positions,
            xticklabels=[str(speaker) for speaker in speakers],
            ylim=(0.0, 1.05),
        )
        speaker_ax.tick_params(axis="x", labelrotation=90, labelsize=8)
        speaker_ax.set_axisbelow(True)
        speaker_ax.grid(axis="y", alpha=0.3)
        _log_figure_with_step(
            speaker_fig, f"{eval_type}_speaker_balanced_accuracy", epoch
        )
        _log_speaker_confident_errors(
            y_true, y_probas, speaker_ids, eval_type, epoch
        )

    _log_probability_histogram(y_true, y_probas, threshold, eval_type, epoch)
    if audio_lengths is not None:
        _log_duration_performance(
            y_true, y_probas >= threshold, audio_lengths, eval_type, epoch
        )
    _log_bac_sensitivity(
        y_true, y_probas >= threshold, bac_values, bac_limit, eval_type, epoch
    )


def _log_speaker_confident_errors(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    speaker_ids: np.ndarray,
    eval_type: str,
    epoch: int | None,
) -> None:
    # Equivalent to |p - y| > 0.9, independent of the decision threshold.
    labels = y_true.astype(bool)
    false_positives = ~labels & (y_probas > 0.9)
    false_negatives = labels & (y_probas < 0.1)
    speakers = np.unique(speaker_ids)
    fp_counts = np.array([
        np.count_nonzero(false_positives & (speaker_ids == speaker))
        for speaker in speakers
    ])
    fn_counts = np.array([
        np.count_nonzero(false_negatives & (speaker_ids == speaker))
        for speaker in speakers
    ])
    totals = fp_counts + fn_counts

    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(speakers)), 4.5), dpi=120)
    positions = np.arange(len(speakers))
    ax.bar(positions, fp_counts, label="Sober: P(intoxicated) > 0.9")
    ax.bar(
        positions, fn_counts, bottom=fp_counts,
        label="Intoxicated: P(intoxicated) < 0.1",
    )
    for position, count in zip(positions, totals):
        ax.text(position, count, str(count), ha="center", va="bottom", fontsize=8)
    ax.set(
        title=f"{eval_type} confident errors by speaker (|p − y| > 0.9)",
        xlabel="Speaker ID", ylabel="Confident error count",
        xticks=positions, xticklabels=[str(speaker) for speaker in speakers],
        ylim=(0.0, max(1, int(totals.max())) * 1.25),
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    _log_figure_with_step(fig, f"{eval_type}_speaker_confident_errors", epoch)


def _log_probability_histogram(
    y_true, y_probas, threshold: float, eval_type: str, epoch: int | None
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    labels = y_true.astype(bool)
    bins = np.linspace(0.0, 1.0, 21)
    for mask, name in ((~labels, "Sober"), (labels, "Intoxicated")):
        ax.hist(
            y_probas[mask], bins=bins, alpha=0.55,
            label=f"{name} (n={mask.sum()})",
        )
    ax.axvline(
        threshold, color="black", linestyle="--",
        label=f"Threshold={threshold:.3f}",
    )
    ax.set(
        title=f"{eval_type} probabilities by true class",
        xlabel="P(intoxicated)", ylabel="Sample count", xlim=(0.0, 1.0),
    )
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    _log_figure_with_step(fig, f"{eval_type}_probability_histogram", epoch)


def _log_duration_performance(
    y_true, y_pred, audio_lengths, eval_type: str, epoch: int | None
) -> None:
    # Fixed bins make evaluations comparable; boundary values enter the next bin.
    bin_ids = np.digitize(audio_lengths, [2.0, 5.0, 10.0, 20.0])
    ranges = ["[0, 2)", "[2, 5)", "[5, 10)", "[10, 20)", "[20, ∞)"]
    labels = y_true.astype(bool)
    scores, tick_labels = [], []
    for index, interval in enumerate(ranges):
        mask = bin_ids == index
        positive = mask & labels
        negative = mask & ~labels
        sensitivity = y_pred[positive].mean() if positive.any() else np.nan
        specificity = (~y_pred[negative]).mean() if negative.any() else np.nan
        scores.append(((sensitivity + specificity) / 2, sensitivity, specificity))
        tick_labels.append(
            f"{interval}\nS={negative.sum()}, I={positive.sum()}"
        )

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    positions = np.arange(len(ranges))
    for column, name in enumerate(("Balanced accuracy", "Sensitivity", "Specificity")):
        values = np.asarray(scores)[:, column]
        offsets = positions + (column - 1) * 0.25
        ax.bar(offsets, values, width=0.25, label=name)
        for position, value in zip(offsets, values):
            if np.isnan(value):
                ax.text(position, 0.02, "N/A", ha="center", rotation=90, fontsize=7)
    ax.set(
        title=f"{eval_type} performance by audio length",
        xlabel="Audio length (s); S=sober, I=intoxicated; N/A=missing class",
        ylabel="Score", ylim=(0.0, 1.2),
        xticks=positions, xticklabels=tick_labels,
    )
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    _log_figure_with_step(fig, f"{eval_type}_audio_length_performance", epoch)


def _log_bac_sensitivity(
    y_true, y_pred, bac_values, bac_limit: float | None,
    eval_type: str, epoch: int | None,
) -> None:
    labels = y_true.astype(bool)
    intoxicated = labels & (bac_values > 0)
    sober = ~labels
    fig, (ax, sober_ax) = plt.subplots(
        1, 2, figsize=(10, 4.5), dpi=120, gridspec_kw={"width_ratios": [4, 1]}
    )
    if intoxicated.any():
        max_bac = max(float(bac_values[intoxicated].max()), bac_limit or 0.0)
        edges = np.round(np.arange(int(np.ceil(max_bac / 0.2)) + 1) * 0.2, 10)
        # Include the cutoff as an edge so that bins cannot straddle it.
        if bac_limit is not None and bac_limit > 0:
            edges = np.unique(np.append(edges, bac_limit))
        bin_ids = np.minimum(
            np.searchsorted(edges, bac_values[intoxicated], side="right") - 1,
            len(edges) - 2,
        )
        predictions = y_pred[intoxicated]
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = bin_ids == index
            count = int(mask.sum())
            score = predictions[mask].mean() if count else np.nan
            center = (left + right) / 2
            ax.bar(center, score, width=(right - left) * 0.9, color="tab:orange")
            ax.text(
                center, score + 0.03 if count else 0.02,
                f"n={count}" if count else "N/A\nn=0",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )
        ax.set_xlim(0.0, edges[-1])
    else:
        ax.text(
            0.5, 0.5, "No intoxicated samples with positive BAC",
            ha="center", va="center", transform=ax.transAxes,
        )
    if bac_limit is not None:
        ax.axvline(
            bac_limit, color="black", linestyle="--",
            label=f"BAC cutoff={bac_limit:g}‰",
        )
        ax.legend(fontsize=8)
    ax.set(
        title="Intoxicated samples (0.2‰ bins; split at cutoff)",
        xlabel="BAC (‰); final bin includes its upper edge",
        ylabel="Sensitivity", ylim=(0.0, 1.3),
    )
    specificity = (~y_pred[sober]).mean() if sober.any() else np.nan
    sober_ax.bar([0], [specificity], color="tab:blue")
    sober_ax.set(
        title="Sober samples", ylabel="Specificity", ylim=(0.0, 1.3),
        xticks=[0], xticklabels=[f"Sober\nn={sober.sum()}"],
    )
    if not sober.any():
        sober_ax.text(0, 0.02, "N/A", ha="center")
    for panel in (ax, sober_ax):
        panel.set_axisbelow(True)
        panel.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{eval_type} detection by BAC range")
    _log_figure_with_step(fig, f"{eval_type}_bac_sensitivity", epoch)


def _log_figure_with_step(fig, image_key: str, epoch: int) -> None:
    fig.tight_layout()
    suffix = f"epoch_{epoch}" if epoch is not None else "final"
    mlflow.log_figure(fig, f"images/{image_key}_{suffix}.png")
    plt.close(fig)
