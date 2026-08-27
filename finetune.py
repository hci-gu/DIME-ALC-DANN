import os
import torch
import mlflow

from model import DANN
from params import Params
from dac218_data import DAC218Data
from utils.argument_parsing import parse_args

# CLI args
args = parse_args()

# User parameters
save_model = args.save_model
p = Params.from_optional_overrides(**vars(args))
if args.max_samples:
    max_samples = args.max_samples
else:
    max_samples = (1000 if p.dev_run else None)
verbose = args.verbose
run_name = args.run_name
SEED = args.seed

# Mlflow tracking
experiment_name = "DANN - finetune"
mlflow.set_experiment(experiment_name)
print(f"Starting Experiment: ### {experiment_name} ###")
print(f"Using MLflow Tracking URI: {mlflow.get_tracking_uri()}")

device = torch.device(p.device)

# load in finetune data
data = DAC218Data(
    max_samples=max_samples,
    seed=SEED,
    verbose=verbose
)

# load in model
model = DANN()
# modify discriminator architecture
model.to(device)


# start finetuning


# save finetuned model
if save_model:
    model.to("cpu")
    run_name = mlflow.active_run().data.tags["mlflow.runName"].replace(" ", "_").replace("/", "_").replace("\\", "_")
    save_path = os.path.join("weights",f"dann_model-{run_name}.pth")
    os.makedirs("weights", exist_ok=True)
    torch.save(model, save_path)
    print(f"Saved model to: {save_path}")