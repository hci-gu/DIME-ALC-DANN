"""Evaluate speaker-specific reconstruction thresholds on ALC utterances."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from io import StringIO
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from utils.argument_parsing import non_negative_int, positive_int

if TYPE_CHECKING:
    from alc_data import ALCData


HIDDEN_DIMENSION = 64
BOTTLENECK_DIMENSION = 8
THRESHOLD_QUANTILE = 0.95


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one sober autoencoder per ALC speaker")
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=non_negative_int, default=1999)
    parser.add_argument("--n-epochs", type=positive_int, default=50)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.seed >= 2**64:
        parser.error("--seed must be smaller than 2**64")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires an available CUDA device")
    return args


def _split_sober_indices(
    indices: list[int], seed: int, speaker_id: int,
) -> tuple[list[int], list[int], list[int]]:
    shuffled = sorted(indices)
    random.Random(f"{seed}:{speaker_id}").shuffle(shuffled)
    n_train = 3 * len(shuffled) // 5
    calibration_end = n_train + len(shuffled) // 5
    return shuffled[:n_train], shuffled[n_train:calibration_end], shuffled[calibration_end:]


def _speaker_features(
    data: ALCData, train_indices: list[int], calibration_indices: list[int],
    test_indices: list[int], device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    data.calculate_mu_sigma(train_indices)
    train, calibration, test = (
        torch.stack([data[index][0] for index in indices]).to(device)
        for indices in (train_indices, calibration_indices, test_indices)
    )
    return train, calibration, test


def _fit_autoencoder(features: Tensor, args: argparse.Namespace) -> nn.Module:
    input_dimension = features.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dimension, HIDDEN_DIMENSION),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIMENSION, BOTTLENECK_DIMENSION),
        nn.Linear(BOTTLENECK_DIMENSION, HIDDEN_DIMENSION),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIMENSION, input_dimension),
    ).to(features.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    for _ in range(args.n_epochs):
        order = torch.randperm(len(features), device=features.device)
        for batch_indices in order.split(args.batch_size):
            batch = features[batch_indices]
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def _reconstruction_errors(model: nn.Module, features: Tensor) -> Tensor:
    return (model(features) - features).square().mean(dim=1)


def _classification_metrics(tp: float, fp: float, tn: float, fn: float) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def _score_errors(
    calibration_errors: Tensor, test_errors: Tensor, n_sober_test: int,
) -> dict[str, int | float]:
    threshold = torch.quantile(
        calibration_errors, THRESHOLD_QUANTILE, interpolation="linear",
    ).item()
    predictions = test_errors > threshold
    fp = int(predictions[:n_sober_test].sum().item())
    tp = int(predictions[n_sober_test:].sum().item())
    counts = {
        "tp": tp,
        "fp": fp,
        "tn": n_sober_test - fp,
        "fn": len(test_errors) - n_sober_test - tp,
    }
    return {"threshold": threshold, **counts, **_classification_metrics(**counts)}


def _aggregate_metrics(rows: list[dict[str, int | float]]) -> dict[str, float]:
    pooled_counts = {name: sum(row[name] for row in rows) for name in ("tp", "fp", "tn", "fn")}
    pooled = _classification_metrics(**pooled_counts)
    return {
        **{f"macro_{name}": sum(row[name] for row in rows) / len(rows) for name in pooled},
        **{f"pooled_{name}": value for name, value in pooled.items()},
    }


def main() -> None:
    args = _parse_args()

    import mlflow
    from tqdm import tqdm

    from alc_data import ALCData

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.manual_seed(args.seed)
    data = ALCData(seed=args.seed)
    sober_label = data.class_mapping["na"]
    intoxicated_label = data.class_mapping["a"]
    by_speaker = defaultdict(lambda: defaultdict(list))
    samples = zip(data.speaker_ids, data.class_labels.tolist(), strict=True)
    for index, (speaker_id, label) in enumerate(samples):
        by_speaker[speaker_id][label].append(index)

    mlflow.set_experiment("autoencoder")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    rows = []
    skipped = []
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            "dataset": "alc",
            "feature_set": data.cache_dict["feature_set"],
            "feature_level": data.cache_dict["feature_level"],
            "sober_split": "60/20/20",
            "threshold_quantile": THRESHOLD_QUANTILE,
            "hidden_dimension": HIDDEN_DIMENSION,
            "bottleneck_dimension": BOTTLENECK_DIMENSION,
            "loss": "MSE",
            "optimizer": "Adam",
            "seed": args.seed,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": str(device),
        })
        if data.files:
            mlflow.log_param("input_dimension", data.tensors_dict[data.files[0]].numel())
        for speaker_id, indices in tqdm(sorted(by_speaker.items()), desc="ALC speakers"):
            sober = indices[sober_label]
            intoxicated = indices[intoxicated_label]
            train_indices, calibration_indices, sober_test = _split_sober_indices(
                sober, args.seed, speaker_id,
            )
            if not (train_indices and calibration_indices and sober_test and intoxicated):
                reason = (
                    "insufficient sober utterances for training, calibration, and testing"
                    if not (train_indices and calibration_indices and sober_test)
                    else "missing intoxicated utterances"
                )
                tqdm.write(f"Skipping speaker {speaker_id}: {reason}")
                skipped.append({
                    "speaker_id": speaker_id,
                    "n_sober": len(sober),
                    "n_intoxicated": len(intoxicated),
                    "reason": reason,
                })
                continue

            train, calibration, test = _speaker_features(
                data, train_indices, calibration_indices, sober_test + intoxicated, device,
            )
            model = _fit_autoencoder(train, args)
            calibration_errors = _reconstruction_errors(model, calibration)
            test_errors = _reconstruction_errors(model, test)
            if not (torch.isfinite(calibration_errors).all() and torch.isfinite(test_errors).all()):
                raise RuntimeError(f"Non-finite reconstruction errors for speaker {speaker_id}")
            rows.append({
                "speaker_id": speaker_id,
                "n_train_sober": len(train_indices),
                "n_calibration_sober": len(calibration_indices),
                "n_test_sober": len(sober_test),
                "n_test_intoxicated": len(intoxicated),
                **_score_errors(calibration_errors, test_errors, len(sober_test)),
            })

        mlflow.log_metrics({"speakers_evaluated": len(rows), "speakers_skipped": len(skipped)})
        if skipped:
            mlflow.log_dict({"speakers": skipped}, "skipped_speakers.json")
        if not rows:
            raise RuntimeError("No ALC speakers have enough sober and intoxicated data for evaluation")

        metrics = _aggregate_metrics(rows)
        mlflow.log_metrics(metrics)
        with StringIO(newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            mlflow.log_text(output.getvalue(), "speaker_metrics.csv")
        print(f"Evaluated {len(rows)} speakers in MLflow run {run.info.run_id}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
