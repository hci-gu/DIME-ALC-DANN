import torch
import mlflow
from train import train, eval
from params import Params
from alc_data import ALCData
from model import DANN
from torch.utils.data import random_split, DataLoader

def main():

    dev_run = True
    save_model = True

    # Mlflow tracking
    mlflow.set_experiment("DANN")
    print(f"Using MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    p = Params(dev_run=dev_run)
    device = torch.device(p.device)

    exit(0)

    # Load data
    data = ALCData()
    train_data, val_data, test_data = random_split(data, [0.8,0.1,0.1])

    # DataLoaders
    train_loader = DataLoader(train_data, p.batch_size, shuffle=True, num_workers=p.n_workers, pin_memory=p.pin_memory)
    val_loader = DataLoader(val_data, p.batch_size, shuffle=False, num_workers=p.n_workers, pin_memory=p.pin_memory)
    test_loader = DataLoader(test_data, p.batch_size, shuffle=False, num_workers=p.n_workers, pin_memory=p.pin_memory)

    # Load model
    model = DANN()

    with mlflow.start_run():

        mlflow.log_params(p)

        # Start training
        train(
            model=model,
            p=p,
            train_loader=train_loader,
            val_loader=val_loader
        )

        # Run evaluation
        test_metrics = eval(model, test_loader, device)
        mlflow.log_metrics(test_metrics)

        # Save model & optimizer
        if save_model:
            model.to("cpu")
            torch.save(model, "dann_model.pth")
        

if __name__ == "__main__":
    main()
