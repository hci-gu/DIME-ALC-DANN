import mlflow
from train import train, eval
from params import Params
from alc_data import ALCData
from model import DANN

def main():

    p = Params()

    # Load data
    data = ALCData()

    # DataLoaders
    train_loader = 1
    val_loader = 1
    test_loader = 1

    # Load model
    model = DANN()

    # Start training
    train(
        model=model,
        p=p,
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Run evaluation
    test_metrics = eval()

    # Save model & optimizer
    

if __name__ == "__main__":
    main()
