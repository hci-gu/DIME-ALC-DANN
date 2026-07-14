import os
import torch
import mlflow
import torch.nn as nn

from time import time
from tqdm import tqdm
from model import MLP
from params import Params
from train import evaluate
from alc_data import ALCData
from utils.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main():

    SEED = 1999
    save_model = True
    torch.manual_seed(SEED)

    p = Params()

    # Mlflow tracking
    experiment_name = "Baseline MLP"
    mlflow.set_experiment(experiment_name)
    print(f"Starting Experiment: ### {experiment_name} ###")
    print(f"Using MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    device = torch.device(p.device)

    # Load data
    print(f"Loading data...")
    data = ALCData(
        max_samples=(1000 if p.dev_run else None),
        seed=SEED,
        verbose=False
    )

    # Train/Val/Test splitting
    train_indices, val_indices, test_indices = data.speaker_split(train_frac=0.7, val_frac=0.15, test_frac=0.15)
    train_data = Subset(data, train_indices)
    val_data = Subset(data, val_indices)
    test_data = Subset(data, test_indices)
    pos_weight = data.calculate_pos_weight(train_indices=train_indices).to(device) if p.use_pos_weight else None
    data.cache(train_indices)

    # DataLoaders
    if p.batch_size < 2:
        raise ValueError("batch_size must be at least two when using BatchNorm")
    if len(train_data) < 2:
        raise ValueError("The training split must contain at least two samples when using BatchNorm")
    drop_last_train = len(train_data) % p.batch_size == 1
    train_loader = DataLoader(train_data, p.batch_size, shuffle=True, num_workers=p.n_workers, pin_memory=p.pin_memory, drop_last=drop_last_train)
    val_loader = DataLoader(val_data, p.batch_size, shuffle=False, num_workers=p.n_workers, pin_memory=p.pin_memory)
    test_loader = DataLoader(test_data, p.batch_size, shuffle=False, num_workers=p.n_workers, pin_memory=p.pin_memory)

    # Load model
    model_config = {
        "input_dimension": 6373,
        "hidden_dimension": 512,
        "output_dimension": 1,
        "p_dropout": 0.5,
    }
    model = MLP(**model_config)
    model.to(device)

    # Optimizer
    optimizer_cls = getattr(torch.optim, p.optimizer)
    optimizer = optimizer_cls(model.parameters(), **p.get_vars_from_prefix("optimizer"))

    # Loss function
    classifier_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with mlflow.start_run():

        # Log model architecture
        architecture = {
            "model_class": type(model).__name__,
            **model_config,
        }
        mlflow.log_params(architecture)
        mlflow.log_dict(
            {**architecture, "layers": str(model)},
            "model_architecture.json",
        )

        # Log data metadata
        mlflow.log_dict(data.get_split_speakers(),"speaker_data_split.json")

        # Start training
        stopping_criterion = EarlyStopping(**p.get_vars_from_prefix("early_stopping"))
        scheduler = ReduceLROnPlateau(optimizer=optimizer, **p.get_vars_from_prefix("scheduler"))
        device = torch.device(p.device)

        print(f"Started training with device: {p.device}")
        t_start = time()
        train_pbar = tqdm(range(p.n_epochs), desc="Training", position=0)
        for epoch in train_pbar:


            # Train pass
            model.train()
            train_loss = 0.0
            for batch_idx, (x,y,_) in enumerate(tqdm(train_loader, desc="[Batch]", position=1, leave=False)):
                if p.dev_run and batch_idx > 3: break
                t_batch_start = time()
                x = x.to(device)
                y = y.to(device, dtype=torch.float32) # class label (intoxicated vs sober)

                class_logits = model(x)

                loss = classifier_loss_fn(class_logits.squeeze(-1), y)

                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t_batch_time = time() - t_batch_start

                global_step = epoch * len(train_loader) + batch_idx
                mlflow.log_metrics({
                    "batch_time": t_batch_time,
                    "train_batch_loss": loss.item(),
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
            train_pbar.set_description_str(f"Training L_tot_tr={train_loss:.4f}|L_clf_val={val_metrics["val/classifier_loss"]:.4f}")

            # Check early stopping
            if stopping_criterion(model, val_metrics[p.optim_metric]):
                break
    

        stopping_criterion.load_best_model(model)
        t_tot = (time() - t_start) / 60.0
        print(f"Finished training after {t_tot:.1f} minutes at epoch {1+epoch}")
        mlflow.log_metric("training_time", t_tot)

        # Run test evaluation
        test_metrics = evaluate(model, p, classifier_loss_fn, test_loader, device, eval_type="test")
        mlflow.log_metrics(test_metrics)

        # Save checkpoint
        if save_model:
            model.to("cpu")
            run_name = mlflow.active_run().data.tags["mlflow.runName"].replace(" ", "_").replace("/", "_").replace("\\", "_")
            save_path = os.path.join("weights",f"mlp-baseline-{run_name}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "model_class": type(model).__name__,
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalization_mean": data.mu.cpu(),
                    "normalization_std": data.sigma.cpu(),
                },
                save_path,
            )
            print(f"Saved model to: {save_path}")
        

if __name__ == "__main__":
    main()
