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

    data.set_mu_sigma(checkpoint["mu"], checkpoint["sigma"]) # Use the stored mu,sigma normalization constants
    test_indices = [
        i for i, speaker in enumerate(data.speaker_ids)
        if speaker in set(data_speaker_split["test_speakers"])
    ]
    train_indices = [
        i for i, speaker in enumerate(data.speaker_ids)
        if speaker in set(data_speaker_split["train_speakers"])
    ]
    test_data = Subset(data, indices=test_indices)
    pos_weight = data.calculate_pos_weight(train_indices=train_indices).to(device) if p.use_pos_weight else None
    eval_loader = DataLoader(test_data, p.batch_size, shuffle=False, num_workers=p.n_workers, pin_memory=p.pin_memory)
    classifier_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    evaluation_results = test_evaluation(
        model=model,
        p=p,
        classifier_loss_fn=classifier_loss_fn,
        eval_loader=eval_loader,
        device=device,
        threshold=checkpoint["threshold"]
    )

    log_name = Path(model_name).stem / ".json"
    mlflow.log_dict(evaluation_results, log_name)
    print(f"Saved the evaluation results to {log_name}")


if __name__ == "__main__":
    model_name = "dann-01c0f372"
    main(model_name=model_name)
