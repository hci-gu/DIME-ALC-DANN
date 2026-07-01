import torch
import mlflow
from time import time
from tqdm import tqdm
from model import DANN
from params import Params
from torch.utils.data import DataLoader
from utils.early_stopping import EarlyStopping
from evaluate import eval


def train(
    model: DANN,
    p: Params,
    train_loader: DataLoader,
    val_loader: DataLoader
    ):

    stopping_criterion = EarlyStopping(p.get_vars_from_prefix("early_stopping"))
    device = torch.device(p.device)


    print(f"Started training with device: {p.device}")
    t_start = time()
    for epoch in range(p.n_epochs):

        # Train pass
        model.train()
        for batch_idx, (x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            loss = 1

            mlflow.log_metric("train_batch_loss", loss)

        # Validation step
        val_metrics = eval(model, val_loader)

        # Log metrics
        mlflow.log_metrics(val_metrics)

        # Check early stopping
        if stopping_criterion(model, val_metrics[p.optim_metric]):
            stopping_criterion.load_best_model(model)
            break

        # LR scheduler etc. 

    

    t_tot = (time() - t_start) / 60.0
    print(f"Finished training after {t_tot:.1f} minutes at epoch {1+epoch}")


@torch.no_grad()
def eval(model: DANN, eval_loader: DataLoader, device) -> dict:
    model.eval()

    for batch_idx, (x,y) in enumerate(eval_loader):
        x = x.to(device)
        y = y.to(device)

        loss = 1
