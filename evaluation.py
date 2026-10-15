import os
import torch
import mlflow
import torch.nn as nn

from model import DANN
from pathlib import Path
from params import Params
from alc_data import ALCData
from train import test_evaluation
from dac218_data import DAC218Data
from utils.argument_parsing import parse_args
from torch.utils.data import DataLoader, Subset

def main(model_name = None):

    # CLI args
    args = parse_args("evaluation")

    # Parse model name
    if args.run_name is not None:
        model_path = os.path.join("weights",args.run_name+".pth")
    else:
        if model_name is not None:
            model_path = os.path.join("weights",model_name+".pth")
        else:
            raise ValueError(f"Either provide a checkpoint via '--run-name' <checkpoint> or specify a model_name")
    

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    data_speaker_split = checkpoint["speaker_splits"]
    p = Params(**checkpoint["params"])
    p.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(p.device)
    model = DANN(p)
    model.to(device)
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

    if args.eval_split == "saved-test":
        if args.data != checkpoint["dataset"]:
            raise ValueError(
                "saved-test requires the training dataset; "
                "use --eval-split all for another dataset"
            )
        test_indices = [
            i for i, speaker in enumerate(data.speaker_ids)
            if speaker in set(data_speaker_split["test_speakers"])
        ]
    else:
        test_indices = list(range(len(data)))

    if not test_indices:
        raise ValueError("The selected evaluation dataset is empty")

    data.set_mu_sigma(checkpoint["mu"], checkpoint["sigma"]) # Use the stored mu,sigma normalization constants
    test_data = Subset(data, indices=test_indices)
    eval_loader = DataLoader(test_data, p.batch_size, shuffle=False, num_workers=p.n_workers, pin_memory=p.pin_memory)

    evaluation_results = test_evaluation(
        model=model,
        p=p,
        eval_loader=eval_loader,
        device=device,
        eval_type="test",
        threshold=checkpoint["threshold"]
    )

    log_name = Path(model_name).stem / ".json"
    mlflow.log_dict(evaluation_results, log_name)
    print(f"Saved the evaluation results to {log_name}")


if __name__ == "__main__":
    model_name = "dann-01c0f372"
    main(model_name=model_name)
