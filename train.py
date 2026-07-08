import torch
import mlflow
import matplotlib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from model import DANN
from params import Params
from typing import Literal
from torch import optim, Tensor
from torch.utils.data import DataLoader
from utils.early_stopping import EarlyStopping
from utils.compute_params import alpha_schedule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve

matplotlib.use("Agg")


def train(
    model: DANN,
    p: Params,
    optimizer: optim,
    loss_functions: tuple[nn.BCEWithLogitsLoss, nn.CrossEntropyLoss],
    train_loader: DataLoader,
    val_loader: DataLoader
    ):

    stopping_criterion = EarlyStopping(**p.get_vars_from_prefix("early_stopping"))
    scheduler = ReduceLROnPlateau(optimizer=optimizer, **p.get_vars_from_prefix("scheduler"))
    device = torch.device(p.device)
    classifier_loss_fn, discriminator_loss_fn = loss_functions # Unpack loss functions


    print(f"Started training with device: {p.device}")
    t_start = time()
    train_pbar = tqdm(range(p.n_epochs), desc="Training", position=0)
    for epoch in train_pbar:

        alpha = alpha_schedule(epoch, p.n_epochs)
        mlflow.log_metric("alpha",alpha, step=epoch)

        # Train pass
        model.train()
        train_loss = 0.0
        for batch_idx, (x,y,s) in enumerate(tqdm(train_loader, desc="[Batch]", position=1, leave=False)):
            if p.dev_run and batch_idx > 3: break
            t_batch_start = time()
            x = x.to(device)
            y = y.to(device, dtype=torch.float32) # class label (intoxicated vs sober)
            s = s.to(device, dtype=torch.long) # speaker local index
            assert (s >= 0).all(), f"Detected negative speaker_id, training must use a training subset"

            class_logits, speaker_logits = model(x, alpha=alpha)

            classifier_loss = classifier_loss_fn(class_logits.squeeze(-1), y)
            discriminator_loss = discriminator_loss_fn(speaker_logits, s)

            loss = classifier_loss + discriminator_loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_batch_time = time() - t_batch_start

            global_step = epoch * len(train_loader) + batch_idx
            mlflow.log_metrics({
                "batch_time": t_batch_time,
                "train_batch_classifier_loss": classifier_loss.item(),
                "train_batch_discriminator_loss": discriminator_loss.item(),
                "train_batch_loss": loss.item()
            }, step=global_step
            )
        train_loss = train_loss / len(train_loader)

        # Validation step
        val_metrics = evaluate(model, p, classifier_loss_fn, val_loader, device, epoch=epoch)

        # Log metrics
        mlflow.log_metrics(val_metrics, step=epoch)

        # LR Scheduler
        scheduler.step(val_metrics[p.optim_metric])

        # Update tqdm bar
        train_pbar.set_description_str(f"Training L_tr={train_loss:.4f}|L_val={val_metrics["val_classifier_loss"]:.4f}")

        # Check early stopping
        if stopping_criterion(model, val_metrics[p.optim_metric]):
            break
    

    stopping_criterion.load_best_model(model)
    t_tot = (time() - t_start) / 60.0
    print(f"Finished training after {t_tot:.1f} minutes at epoch {1+epoch}")
    mlflow.log_metric("training_time", t_tot)


@torch.no_grad()
def evaluate(model: DANN, p:Params, classifier_loss_fn, eval_loader: DataLoader, device, epoch: int = None, eval_type: Literal["val","test"] = "val") -> dict:
    model.eval()

    total_classifier_loss = 0.0
    n_correct = 0
    fp, fn, tp, tn = 0, 0, 0, 0
    y_true, y_probas = [], []
    for (x,y,_) in tqdm(eval_loader, desc="[Validation]", position=1, leave=False):
        x: Tensor = x.to(device) # [B,d_input]
        y = y.to(device) # class label (intoxicated vs sober)

        class_logits = model.predict(x)
        y_prob = torch.sigmoid(class_logits.squeeze(-1))
        y = y.bool()

        # Loss
        total_classifier_loss += classifier_loss_fn(class_logits.squeeze(-1), y.to(torch.float32)).item()

        y_true.append(y.cpu().numpy())
        y_probas.append(y_prob.cpu().numpy())
    total_classifier_loss = total_classifier_loss / len(eval_loader)
    y_true = np.concatenate(y_true)
    y_probas = np.concatenate(y_probas)

    # Threshold probabilities to get vector
    y_pred = (y_probas > 0.5)
    
    # Confusion matrix elements
    tp = (y_pred & y_true).sum()
    tn = (~y_pred & ~y_true).sum()
    fp = (y_pred & ~y_true).sum()
    fn = (~y_pred & y_true).sum()

    # Accuracy, P, R, Specificity, F1
    n_correct = (y_pred == y_true).sum()
    accuracy = n_correct/len(eval_loader.dataset)
    precision = tp / (tp + fp) if (tp + fp > 0) else 0.0
    recall = tp / (tp + fn) if (tp + fn > 0) else 0.0
    specificity = tn / (tn + fp) if (tn + fp > 0) else 0.0
    balanced_accuracy = (recall + specificity) / 2
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall > 0) else 0.0

    has_both_classes = len(set(y_true.tolist())) == 2
    auroc = roc_auc_score(y_true, y_probas) if has_both_classes else 0.0

    # Precision-Recall Curve
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_probas)
    pr_f1 = 2 * pr_precision[:-1] * pr_recall[:-1] / (pr_precision[:-1] + pr_recall[:-1] + 1e-12)
    best_f1 = pr_f1.max()
    best_threshold = round(float(pr_thresholds[pr_f1.argmax()]), 4) if len(pr_thresholds) else 0.5

    # Log figures every 5th epoch or on test set
    if (eval_type == "test") or ((epoch % 5 == 0) if epoch else False):
        _log_classifier_curves(
            y_true=y_true,
            y_probas=y_probas,
            pr_precision=pr_precision,
            pr_recall=pr_recall,
            auroc=auroc,
            has_both_classes=has_both_classes,
            eval_type=eval_type,
            epoch=epoch,
        )

    
    # Dictionary containing the metrics
    metrics =  {
        "classifier_loss": total_classifier_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "balanced_accuracy": balanced_accuracy,
        "specificity": specificity,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    return {f"{eval_type}_{key}": value for (key,value) in metrics.items()}



### HELPER FUNCTIONS ### 

def _log_classifier_curves(
    y_true,
    y_probas,
    pr_precision,
    pr_recall,
    auroc: float,
    has_both_classes: bool,
    eval_type: str,
    epoch: int,
) -> None:
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


def _log_figure_with_step(fig, image_key: str, epoch: int) -> None:
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    image = Image.open(buffer).copy()
    plt.close(fig)

    try:
        mlflow.log_image(image, key=image_key, step=epoch)
    except TypeError:
        mlflow.log_image(image, artifact_file=f"{image_key}_epoch_{epoch:04d}.png")

