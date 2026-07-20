import torch
import mlflow
import matplotlib
import numpy as np
import torch.nn as nn
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm
from model import DANN
from optuna import Trial
from params import Params
from typing import Literal
from functools import partial
from dataclasses import asdict
from torch import optim, Tensor
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from utils.early_stopping import EarlyStopping
from utils.compute_params import alpha_schedule
from utils.focal_loss import MulticlassFocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve



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
        n_examples = 0
        train_loss = 0.0
        n_correct_classifier = 0.0
        n_correct_discriminator = 0.0
        for batch_idx, (x,y,s) in enumerate(tqdm(train_loader, desc="[Batch]", position=1, leave=False)):
            if p.dev_run and batch_idx > 3: break

            t_batch_start = time()
            x = x.to(device)
            y = y.to(device, dtype=torch.float32) # class label (intoxicated vs sober)
            s = s.to(device, dtype=torch.long) # speaker local index
            assert (s >= 0).all(), f"Detected negative speaker_id, training must use a training subset"

            class_logits, speaker_logits = model(x, alpha=alpha)

            # Classifier and Discriminator accuracy
            n_correct_classifier += ((class_logits.squeeze(-1) >= 0.0) == y.to(torch.bool)).sum().item() # Naive 0.5 sigmoid-threshold 
            n_correct_discriminator += (speaker_logits.argmax(dim=1) == s).sum().item()
            n_examples += y.numel()

            # Classifier and Discriminator loss
            classifier_loss = classifier_loss_fn(class_logits.squeeze(-1), y)
            discriminator_loss = discriminator_loss_fn(speaker_logits, s)

            loss = classifier_loss + discriminator_loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_batch_time = time() - t_batch_start

            global_step = epoch * len(train_loader) + batch_idx
            mlflow.log_metrics(
                {
                "batch_time": t_batch_time,
                "train_batch_classifier_loss": classifier_loss.item(),
                "train_batch_discriminator_loss": discriminator_loss.item(),
                "train_batch_loss": loss.item()
                },
                step=global_step
            )
        train_loss = train_loss / len(train_loader)
        classifier_accuracy = n_correct_classifier / n_examples
        discriminator_accuracy = n_correct_discriminator / n_examples
        mlflow.log_metrics(
            {
            "train_loss": train_loss,
            "classifier_accuracy": classifier_accuracy,
            "discriminator_accuracy": discriminator_accuracy,
            "discriminator_chance_delta": discriminator_accuracy - 1/p.discriminator_output_dimension # deviation from random chance
            },
            step=epoch
        )

        # Train & Validation evaluation
        val_metrics = evaluate(model, p, classifier_loss_fn, val_loader, device, epoch=epoch, eval_type="val")

        # Log metrics
        mlflow.log_metrics(val_metrics, step=epoch)

        # LR Scheduler
        scheduler.step(val_metrics[p.optim_metric])

        # Update tqdm bar
        train_pbar.set_description_str(f"Training L_tot_tr={train_loss:.4f}|L_clf_val={val_metrics["val/classifier_loss"]:.4f}")

        # Check early stopping
        if stopping_criterion(model, val_metrics[p.optim_metric]):
            break
    

    stopping_criterion.load_best_model(model)
    t_tot = (time() - t_start) / 60.0
    print(f"Finished training after {t_tot:.1f} minutes at epoch {1+epoch}")
    mlflow.log_metric("training_time_minutes", t_tot)


@torch.no_grad()
def evaluate(model: nn.Module, p:Params, classifier_loss_fn, eval_loader: DataLoader, device, epoch: int = None, eval_type: Literal["val","test"] = "val") -> dict:
    model.eval()

    total_classifier_loss = 0.0
    n_correct = 0
    fp, fn, tp, tn = 0, 0, 0, 0
    y_true, y_probas = [], []
    for (x,y,_) in tqdm(eval_loader, desc="[Validation]", position=1, leave=False):
        x: Tensor = x.to(device) # [B,d_input]
        y = y.to(device) # class label (intoxicated vs sober)

        class_logits = model.predict(x) if hasattr(model, "predict") else model(x)
        y_prob = torch.sigmoid(class_logits.squeeze(-1))
        y = y.bool()

        # Loss
        total_classifier_loss += classifier_loss_fn(class_logits.squeeze(-1), y.to(torch.float32)).item()

        y_true.append(y.cpu().numpy())
        y_probas.append(y_prob.cpu().numpy())
    total_classifier_loss = total_classifier_loss / len(eval_loader)
    y_true = np.concatenate(y_true)
    y_probas = np.concatenate(y_probas)

    # Precision-Recall Curve & optimal threshold
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_probas)
    pr_f1 = 2 * pr_precision[:-1] * pr_recall[:-1] / (pr_precision[:-1] + pr_recall[:-1] + 1e-12)
    best_f1 = pr_f1.max()
    best_threshold = float(pr_thresholds[pr_f1.argmax()]) if len(pr_thresholds) else 0.5

    # Threshold probabilities to get vector
    y_pred = (y_probas >= best_threshold)
    
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

    # Log figures every 5th epoch or on final test set
    if (eval_type == "test") or (epoch is not None and epoch % 5 == 0):
        log_evaluation_figures(
            y_true=y_true,
            y_probas=y_probas,
            pr_precision=pr_precision,
            pr_recall=pr_recall,
            confusion_matrix=(tp, tn, fp, fn),
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
        "best_threshold": round(best_threshold, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    return {f"{eval_type}/{key}": value for (key,value) in metrics.items()}


def objective(trial: Trial, train_data, val_data, d_discriminator: int, pos_weight):

    # HPO parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 24, 32])
    classifier_loss_fn_str = trial.suggest_categorical("classifier_loss_fn", ["BCEWithLogitsLoss", "FocalLoss"])  
    discriminator_loss_fn_str = trial.suggest_categorical("discriminator_loss_fn", ["CrossEntropyLoss", "FocalLoss"])  
    optimizer_str = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "SGD", "RMSprop"])

    # Param class
    p = Params(
        batch_size=batch_size,
        optimizer_lr=learning_rate,
        discriminator_output_dimension=d_discriminator
    )

    device = torch.device(p.device)

    # DataLoaders
    train_loader = DataLoader(train_data, p.batch_size, shuffle=True, num_workers=p.n_workers, pin_memory=p.pin_memory)
    val_loader = DataLoader(val_data, p.batch_size, shuffle=False, num_workers=p.n_workers, pin_memory=p.pin_memory)

    # Load model
    model = DANN(p)
    model.to(device)

    # Optimizer
    if optimizer_str == "AdamW":
        betas=(
            trial.suggest_float("adamw_beta1", 0.85, 0.95),
            trial.suggest_float("adamw_beta2", 0.95, 0.9999),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas
        )
    elif optimizer_str == "Adam":
        betas=(
            trial.suggest_float("adam_beta1", 0.85, 0.95),
            trial.suggest_float("adam_beta2", 0.95, 0.9999),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas
        )
    elif optimizer_str == "SGD":
        momentum = trial.suggest_float("sgd_momentum", 0.8, 0.99)
        nesterov = trial.suggest_categorical("sgd_nesterov", [True, False])
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov
        )
    elif optimizer_str == "RMSprop":
        alpha = trial.suggest_float("rmsprop_alpha", 0.9, 0.999)
        momentum = trial.suggest_float("rmsprop_momentum", 0, 0.8)
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            alpha=alpha,
            momentum=momentum
        )
    else:
        raise ValueError(f"Unexpected optimizer value: {optimizer_str}")


    # Classifier Loss Function
    if classifier_loss_fn_str == "BCEWithLogitsLoss":
        use_pos_weight = trial.suggest_categorical("use_pos_weight", [True, False])
        pos_weight = pos_weight.to(device)
        classifier_loss_fn = nn.BCEWithLogitsLoss(pos_weight=(pos_weight if use_pos_weight else None))
    elif classifier_loss_fn_str == "FocalLoss":
        classifier_alpha = trial.suggest_float("classifier_alpha", 0.05, 0.9)
        classifier_gamma = trial.suggest_float("classifier_gamma", 1.0, 3.0)
        classifier_loss_fn = partial(sigmoid_focal_loss, alpha=classifier_alpha, gamma=classifier_gamma, reduction="mean")
    
    # Discriminator Loss Function
    if discriminator_loss_fn_str == "CrossEntropyLoss":
        label_smoothing = trial.suggest_float("label_smoothing", 0, 0.2)
        discriminator_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif discriminator_loss_fn_str == "FocalLoss":
        discriminator_gamma = trial.suggest_float("discriminator_gamma", 1.0, 3.0)
        discriminator_loss_fn = MulticlassFocalLoss(gamma=discriminator_gamma)
    loss_functions = (classifier_loss_fn, discriminator_loss_fn)

    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):

        # Log trial params
        merged_p = asdict(p) | trial.params # Overwrite with trial parameters
        mlflow.log_params(merged_p)
        
        train(
            model=model,
            p=p,
            optimizer=optimizer,
            loss_functions=loss_functions,
            train_loader=train_loader,
            val_loader=val_loader
        )

        # evaluate model
        val_metrics = evaluate(model, p, classifier_loss_fn, val_loader, device, eval_type="val")

        mlflow.log_metrics(val_metrics)
        mlflow.log_metric("hpo/objective", val_metrics[p.optim_metric])

    return val_metrics[p.optim_metric]




### HELPER FUNCTIONS ### 

def log_evaluation_figures(
    y_true,
    y_probas,
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

    # Reciever Operating Characteristic (ROC) curve
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
    suffix = f"epoch_{epoch}" if epoch is not None else "final"
    mlflow.log_figure(fig, f"images/{image_key}_{suffix}.png")
    plt.close(fig)
