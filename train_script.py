import mlflow
from train import train
from alc_data import ALCData
from model import DANN

def main():

    # Load data
    data = ALCData()

    # DataLoaders

    # Load model
    model = DANN()

    # Start training
    train()

    # Run evaluation

    # Save model & optimizer
    

if __name__ == "__main__":
    main()
