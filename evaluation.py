import os
import torch

from model import DANN
from params import Params
from alc_data import ALCData
from dac218_data import DAC218Data
from utils.argument_parsing import parse_args
from train import evaluate

def main(model_name = None):

    # CLI args
    args = parse_args("evaluation")


    # Parse model name
    if model_name:
        model_path = os.path.join("weights",model_name+".pth")
    else:
        if args.run_name:
            model_path = os.path.join("weights",args.run_name+".pth")
        else:
            raise ValueError(f"Provide either a model name ")
    

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    p = Params(**checkpoint["params"])
    model = DANN(p)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.compile:
        model = torch.compile(model)

    # Load in data
    if args.data.lower() == "alc":
        data = ALCData(
            seed=args.seed,
            lower_bac_limit=args.bac_limit,
            verbose=args.verbose
        )
    elif args.data.lower() == "dac":
        data = DAC218Data(
            seed=args.seed,
            lower_bac_limit=args.bac_limit,
            verbose=args.verbose
        )
    else:
        raise RuntimeError(f"Unexpected data type {args.data}")




    

if __name__ == "__main__":
    model_name = "dann-01c0f372"
    main(model_name=model_name)
