import os
import torch
import mlflow
import torch.nn as nn

from model import DANN
from params import Params
from dataclasses import asdict
from train import train, evaluate
from dac218_data import DAC218Data
from utils.argument_parsing import parse_args
from torch.utils.data import DataLoader, Subset

# CLI args
args = parse_args(profile="finetune")

# User parameters
save_model = args.save_model
p = Params.from_optional_overrides(**vars(args))
if args.max_samples:
    max_samples = args.max_samples
else:
    max_samples = (1000 if p.dev_run else None)
if args.checkpoint:
    checkpoint_name = args.checkpoint
else:
    checkpoint_name = "dann_model-unequaled-rat-371.pth"

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

# Train/Val/Test splitting
train_indices, val_indices, test_indices = data.speaker_split(train_frac=0.7, val_frac=0.15, test_frac=0.15)
train_data = Subset(data, train_indices)
val_data = Subset(data, val_indices)
test_data = Subset(data, test_indices)
p.discriminator_output_dimension = len(data.train_speakers_id) # n_speakers in train_data
pos_weight = data.calculate_pos_weight(train_indices=train_indices).to(device) if p.use_pos_weight else None
data.cache(train_indices)

# DataLoaders
train_loader = DataLoader(train_data, p.batch_size, shuffle=True, num_workers=p.n_workers, pin_memory=p.pin_memory)
val_loader = DataLoader(val_data, p.batch_size, shuffle=False, num_workers=p.n_workers, pin_memory=p.pin_memory)
test_loader = DataLoader(test_data, p.batch_size, shuffle=False, num_workers=p.n_workers, pin_memory=p.pin_memory)


# Load in pre-trained model
model = DANN(p)
pretrained_model: DANN = torch.load(os.path.join("weights", checkpoint_name), map_location="cpu", weights_only=False)
model.extractor.load_state_dict(pretrained_model.extractor.state_dict())
model.classifier.load_state_dict(pretrained_model.classifier.state_dict())
model.to(device)

# Optimizer
optimizer_cls = getattr(torch.optim, p.optimizer)
#optimizer = optimizer_cls(model.parameters(), **p.get_vars_from_prefix("optimizer"))
optimizer = torch.optim.RMSprop([
    {"params": model.extractor.parameters(), "lr": 1e-5},
    {"params": model.classifier.parameters(), "lr": 5e-5},
    {"params": model.discriminator.parameters(), "lr": 5e-5},
])

# Loss function
classifier_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
discriminator_loss_fn = nn.CrossEntropyLoss()
loss_functions = (classifier_loss_fn, discriminator_loss_fn)


# start finetuning
with mlflow.start_run(run_name=run_name, tags={"run_type": "finetune"}):

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

    # Run test evaluation
    test_metrics = evaluate(model, p, classifier_loss_fn, test_loader, device, eval_type="test")
    mlflow.log_metrics(test_metrics)

    # Save finetuned model
    if save_model:
        model.to("cpu")
        run_name = mlflow.active_run().data.tags["mlflow.runName"].replace(" ", "_").replace("/", "_").replace("\\", "_")
        save_path = os.path.join("weights",f"dann_model-{run_name}.pth")
        os.makedirs("weights", exist_ok=True)
        torch.save(model, save_path)
        print(f"Saved model to: {save_path}")