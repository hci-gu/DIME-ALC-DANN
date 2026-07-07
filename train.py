import torch
import mlflow
import torch.nn as nn

from time import time
from tqdm import tqdm
from model import DANN
from torch import optim
from params import Params
from torch.utils.data import DataLoader
from utils.early_stopping import EarlyStopping
from utils.compute_params import alpha_schedule
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
            assert s >= 0, f"Detected negative speaker_id, training must use a training subset"

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
        val_metrics = evaluate(model, p, classifier_loss_fn, val_loader, device)

        # Log metrics
        mlflow.log_metrics(val_metrics, step=epoch)

        # LR Scheduler
        scheduler.step(val_metrics[p.optim_metric])

        # Update tqdm bar
        train_pbar.set_description_str(f"Training L_tr={train_loss:.4f}|L_val={val_metrics["classifier_loss"]:.4f}")

        # Check early stopping
        if stopping_criterion(model, val_metrics[p.optim_metric]):
            break
    

    stopping_criterion.load_best_model(model)
    t_tot = (time() - t_start) / 60.0
    print(f"Finished training after {t_tot:.1f} minutes at epoch {1+epoch}")
    mlflow.log_metric("training_time", t_tot)


# TODO add more classifier metrics here
@torch.no_grad()
def evaluate(model: DANN, p:Params, classifier_loss_fn, eval_loader: DataLoader, device) -> dict:
    model.eval()

    total_classifier_loss = 0.0
    for (x,y,_) in tqdm(eval_loader, desc="[Validation]", position=1, leave=False):
        x = x.to(device)
        y = y.to(device, dtype=torch.float32) # class label (intoxicated vs sober)

        class_logits, _ = model(x, alpha=1.0)

        total_classifier_loss += classifier_loss_fn(class_logits.squeeze(-1), y)

    total_classifier_loss = total_classifier_loss / len(eval_loader)

    return {
        "classifier_loss": total_classifier_loss.item(),
    }

