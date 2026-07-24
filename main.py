import os
import torch
import optuna
import mlflow
import torch.nn as nn

from model import DANN
from params import Params
from alc_data import ALCData
from functools import partial
from dataclasses import asdict
from utils.hpo_status import filter_study
from train import train, evaluate, objective
from utils.argument_parsing import parse_args
from torch.utils.data import DataLoader, Subset


def main():

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
    SEED = args.seed

    # Mlflow tracking
    experiment_name = "DANN"
    mlflow.set_experiment(experiment_name)
    print(f"Starting Experiment: ### {experiment_name} ###")
    print(f"Using MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    device = torch.device(p.device)

    # Load data
    print(f"Loading data...")
    data = ALCData(
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

    if args.hpo: # Perform HPO

        with mlflow.start_run(run_name="HPO", tags={"run_type": "hpo"}):

            # HPO parameters
            N_TRIALS = 100
            N_WARMUP_TRIALS = 10
            TIMEOUT_IN_SECONDS = int(60 * 60 * 24 * 4.0)  # 4 Days in seconds

            mlflow.log_params({
                "optim_metric": p.optim_metric,
                "dev_run": p.dev_run,
                "n_trials": N_TRIALS,
                "n_warmup_trials": N_WARMUP_TRIALS,
                "timeout": TIMEOUT_IN_SECONDS,
                "optuna_seed": SEED
            })

            # Log data metadata
            mlflow.log_dict(data.get_split_speakers(),"speaker_data_split.json")

            # Perform HPO
            sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_WARMUP_TRIALS, multivariate=True)
            study = optuna.create_study(direction="maximize", sampler=sampler, study_name="dann_alc")
            objective_fn = partial(objective, train_data=train_data, val_data=val_data, base_params=p, pos_weight=pos_weight)
            study.optimize(objective_fn, n_trials=N_TRIALS, timeout=TIMEOUT_IN_SECONDS, catch=(torch.cuda.OutOfMemoryError))

            # Analyze study trials
            completed, failed, pruned = filter_study(study)

            mlflow.log_metrics({
                "hpo/completed_trials": len(completed),
                "hpo/failed_trials": len(failed),
                "hpo/pruned_trials": len(pruned),
                "hpo/total_trials": len(study.trials),
            })

            if not completed:
                raise RuntimeError(
                    "HPO finished without a completed trial; "
                    "see the failed child runs for the underlying errors."
                )

            # Select the best trial parameters
            best_trial = study.best_trial

            mlflow.log_metric("hpo/best_trial_number", best_trial.number)
            mlflow.log_metric("hpo/best_objective", float(best_trial.value))
            mlflow.log_params({
                f"hpo/best_{name}": value
                for name, value in best_trial.params.items()
            })
            mlflow.log_dict(
                {
                "trial_number": best_trial.number,
                "objective_value": best_trial.value,
                "params": best_trial.params,
                },
                "hpo/best_trial.json",
            )

    else: # Regular training

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

            # Run test evaluation
            test_metrics = evaluate(model, p, classifier_loss_fn, test_loader, device, eval_type="test")
            mlflow.log_metrics(test_metrics)

            # Save model & optimizer
            if save_model:
                model.to("cpu")
                run_name = mlflow.active_run().data.tags["mlflow.runName"].replace(" ", "_").replace("/", "_").replace("\\", "_")
                save_path = os.path.join("weights",f"dann_model-{run_name}.pth")
                os.makedirs("weights", exist_ok=True)
                torch.save(model, save_path)
                print(f"Saved model to: {save_path}")
        

if __name__ == "__main__":
    main()
