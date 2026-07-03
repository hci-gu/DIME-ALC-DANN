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

    stopping_criterion = EarlyStopping(p.get_vars_from_prefix("early_stopping"))
    scheduler = ReduceLROnPlateau(optimizer=optimizer, **p.get_vars_from_prefix("scheduler"))
    device = torch.device(p.device)
    classifier_loss_fn, discriminator_loss_fn = loss_functions # Unpack loss functions


    print(f"Started training with device: {p.device}")
    t_start = time()
    for epoch in tqdm(range(p.n_epochs)):

        alpha = alpha_schedule(epoch, p.n_epochs)
        mlflow.log_metric("alpha",alpha, step=epoch)

        # Train pass
        model.train()
        for batch_idx, (x,y,s) in enumerate(train_loader):
            t_batch_start = time()
            x = x.to(device)
            y = y.to(device, dtype=torch.float32) # class label (intoxicated vs sober)
            s = s.to(device, dtype=torch.long) # speaker ID 

            class_logits, speaker_logits = model(x, alpha=alpha)

            classifier_loss = classifier_loss_fn(class_logits.squeeze(-1), y)
            discriminator_loss = discriminator_loss_fn(speaker_logits, s)

            loss = classifier_loss + discriminator_loss

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

        # Validation step
        val_metrics = evaluate(model, p, loss_functions, val_loader, device)

        # Log metrics
        mlflow.log_metrics(val_metrics, step=epoch)

        # LR Scheduler
        scheduler.step(val_metrics[p.optim_metric])

        # Check early stopping
        if stopping_criterion(model, val_metrics[p.optim_metric]):
            break
    

    t_tot = (time() - t_start) / 60.0
    print(f"Finished training after {t_tot:.1f} minutes at epoch {1+epoch}")
    mlflow.log_metric("training_time", t_tot)


@torch.no_grad()
def evaluate(model: DANN, p:Params, loss_functions, eval_loader: DataLoader, device) -> dict:
    model.eval()

    classifier_loss_fn, discriminator_loss_fn = loss_functions

    total_classifier_loss = 0.0
    total_discriminator_loss = 0.0
    for (x,y,s) in eval_loader:
        x = x.to(device)
        y = y.to(device, dtype=torch.float32) # class label (intoxicated vs sober)
        s = s.to(device, dtype=torch.long) # speaker ID 

        class_logits, speaker_logits = model(x, alpha=1.0)

        total_classifier_loss += classifier_loss_fn(class_logits.squeeze(-1), y)
        total_discriminator_loss += discriminator_loss_fn(speaker_logits, s)

    total_classifier_loss = total_classifier_loss / len(eval_loader)
    total_discriminator_loss = total_discriminator_loss / len(eval_loader)

    return {
        "classifier_loss": total_classifier_loss.item(),
        "discriminator_loss": total_discriminator_loss.item(),
        "loss": (total_classifier_loss + total_discriminator_loss).item()
    }

