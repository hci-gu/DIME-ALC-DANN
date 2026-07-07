import torch
import mlflow
import torch.nn as nn

from model import DANN
from params import Params
from alc_data import ALCData
from dataclasses import asdict
from train import train, evaluate
from utils.argument_parsing import parse_args
from torch.utils.data import DataLoader, Subset


def main():

    # CLI args
    args = parse_args()

    # User parameters
    save_model = args.save_model
    p = Params.from_optional_overrides(**vars(args))
    max_samples = args.max_samples if (args.max_samples is not None) else (1000 if p.dev_run else None)
    verbose = args.verbose
    SEED = 1999

    # Mlflow tracking
    mlflow.set_experiment("DANN"+ " - DEV" if p.dev_run else "")
    print(f"Using MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    device = torch.device(p.device)

    # Load data
    print(f"Loading data...")
    data = ALCData(
        max_samples=max_samples,
        cache_features=args.cache_features,
        seed=SEED,
        verbose=verbose
    )
    train_indices, val_indices, test_indices = data.speaker_split(train_frac=0.8, val_frac=0.1, test_frac=0.1)
    pos_weight = data.calculate_pos_weight(train_indices=train_indices).to(device) if p.use_pos_weight else None
    p.discriminator_output_dimension = len(data.speaker_id_to_index) # n_speakers

    # Train/Val/Test splitting
    train_data = Subset(data, train_indices)
    val_data = Subset(data, val_indices)
    test_data = Subset(data, test_indices)

    # DataLoaders
    train_loader = DataLoader(train_data, p.batch_size, shuffle=True, num_workers=p.n_workers, pin_memory=p.pin_memory)
    val_loader = DataLoader(val_data, p.batch_size, shuffle=False, num_workers=p.n_workers, pin_memory=p.pin_memory)
    test_loader = DataLoader(test_data, p.batch_size, shuffle=False, num_workers=p.n_workers, pin_memory=p.pin_memory)

    # Load model
    model = DANN(p)
    model.to(device)

    # Optimizer
    optimizer_cls = getattr(torch.optim, p.optimizer)
    optimizer = optimizer_cls(model.parameters(), **p.get_vars_from_prefix("optimizer"))

    # Loss function
    classifier_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    discriminator_loss_fn = nn.CrossEntropyLoss()
    loss_functions = (classifier_loss_fn, discriminator_loss_fn)

    with mlflow.start_run():

        # Log parameters
        mlflow.log_params(asdict(p))

        # Log data metadata
        mlflow.log_dict(data.get_split_speakers(),"speaker_data_split.json")

        # Start training
        train(
            model=model,
            p=p,
            optimizer=optimizer,
            loss_functions=loss_functions,
            train_loader=train_loader,
            val_loader=val_loader
        )

        # Run evaluation
        test_metrics = evaluate(model, p, loss_functions, test_loader, device)
        mlflow.log_metrics(test_metrics)

        # Save model & optimizer
        if save_model:
            model.to("cpu")
            torch.save(model, "dann_model.pth")
        

if __name__ == "__main__":
    main()
