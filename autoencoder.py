"""Evaluate speaker-specific intoxication scores on ALC utterances."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
SWEEP_QUANTILES = (0.80, 0.85, 0.90, THRESHOLD_QUANTILE)


@dataclass
class SpeakerSplit:
    train: list[int]
    calibration: list[int]
    sober_test: list[int]
    intoxicated: list[int]
    intoxicated_enrollment: list[int] = field(default_factory=list)

    @property
    def test(self) -> list[int]:
        return self.sober_test + self.intoxicated


@dataclass
class SpeakerScores:
    calibration: Tensor
    test: Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate speaker-specific intoxication models on ALC")
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=non_negative_int, default=1999)
    parser.add_argument("--n-epochs", type=positive_int, default=50)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--method", choices=("mlp", "pca", "shared-mlp", "logistic"), default="mlp")
    parser.add_argument("--feature-set", choices=("ComParE_2016", "eGeMAPSv02"), default="ComParE_2016")
    parser.add_argument("--threshold-sweep", action="store_true", help="Select a cutoff on development speakers")
    parser.add_argument("--target-fpr", type=float, default=0.15, help="Maximum development sober false-positive rate")
    parser.add_argument("--pca-components", type=positive_int, default=BOTTLENECK_DIMENSION)
    parser.add_argument("--lr-c", type=float, default=0.1, help="Inverse L2 regularization strength for logistic regression")
    parser.add_argument(
        "--reserve-intoxicated-training", action="store_true",
        help="Use the same 40%% intoxicated test subset as logistic regression",
    )
    args = parser.parse_args()
    if args.seed >= 2**64:
        parser.error("--seed must be smaller than 2**64")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.lr_c) or args.lr_c <= 0:
        parser.error("--lr-c must be finite and positive")
    if args.method == "logistic" and args.device == "cuda":
        parser.error("--method logistic uses CPU. Choose --device cpu or auto")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires an available CUDA device")
    if not 0 <= args.target_fpr <= 1:
        parser.error("--target-fpr must be between 0 and 1")
    if args.method == "shared-mlp" and not args.threshold_sweep:
        parser.error("--method shared-mlp requires --threshold-sweep for speaker-disjoint evaluation")
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
        _normalized_features(data, indices).to(device)
        for indices in (train_indices, calibration_indices, test_indices)
    )
    return train, calibration, test


def _normalized_features(data: ALCData, indices: list[int]) -> Tensor:
    return torch.stack([data[index][0] for index in indices])


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


def _fit_pca(features: Tensor, n_components: int) -> Callable[[Tensor], Tensor]:
    mean = features.mean(dim=0)
    centered = features - mean
    _, singular_values, vectors = torch.linalg.svd(centered, full_matrices=False)
    tolerance = torch.finfo(features.dtype).eps * max(centered.shape) * singular_values[0]
    rank = int((singular_values > tolerance).sum().item())
    basis = vectors[:min(n_components, len(features) - 1, rank)]

    def reconstruct(inputs: Tensor) -> Tensor:
        return ((inputs - mean) @ basis.T) @ basis + mean

    return reconstruct


@torch.inference_mode()
def _reconstruction_errors(model: Callable[[Tensor], Tensor], features: Tensor) -> Tensor:
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
    quantile: float | None = THRESHOLD_QUANTILE,
) -> dict[str, int | float]:
    threshold = math.inf if quantile is None else torch.quantile(
        calibration_errors, quantile, interpolation="linear",
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


def _speaker_splits(
    data: ALCData, seed: int, *, reserve_intoxicated_training: bool = False,
) -> tuple[dict[int, SpeakerSplit], list[dict[str, int | str]]]:
    by_speaker = defaultdict(lambda: defaultdict(list))
    for index, (speaker_id, label) in enumerate(zip(data.speaker_ids, data.class_labels.tolist(), strict=True)):
        by_speaker[speaker_id][label].append(index)

    splits = {}
    skipped = []
    minimum_intoxicated = 2 if reserve_intoxicated_training else 1
    for speaker_id, indices in sorted(by_speaker.items()):
        sober = indices[data.class_mapping["na"]]
        intoxicated = indices[data.class_mapping["a"]]
        train, calibration, test = _split_sober_indices(sober, seed, speaker_id)
        if not (train and calibration and test) or len(intoxicated) < minimum_intoxicated:
            skipped.append({
                "speaker_id": speaker_id,
                "n_sober": len(sober),
                "n_intoxicated": len(intoxicated),
                "reason": "insufficient sober utterances" if len(sober) < 5 else "insufficient intoxicated utterances",
            })
            continue
        intoxicated_enrollment = []
        if reserve_intoxicated_training:
            intoxicated = sorted(intoxicated)
            random.Random(f"{seed}:{speaker_id}:intoxicated").shuffle(intoxicated)
            n_train = 3 * len(intoxicated) // 5
            intoxicated_enrollment, intoxicated = intoxicated[:n_train], intoxicated[n_train:]
        splits[speaker_id] = SpeakerSplit(train, calibration, test, intoxicated, intoxicated_enrollment)
    return splits, skipped


def _speaker_partitions(speaker_ids: Sequence[int], seed: int) -> dict[str, list[int]]:
    if len(speaker_ids) < 3:
        raise ValueError("The threshold sweep requires at least three eligible ALC speakers")
    shuffled = sorted(speaker_ids)
    random.Random(seed).shuffle(shuffled)
    train_end = 3 * len(shuffled) // 5
    dev_end = train_end + max(1, len(shuffled) // 5)
    return {
        "shared_training": sorted(shuffled[:train_end]),
        "development": sorted(shuffled[train_end:dev_end]),
        "held_out": sorted(shuffled[dev_end:]),
    }


def _fit_shared_autoencoder(
    data: ALCData, splits: dict[int, SpeakerSplit], training_speakers: list[int],
    device: torch.device, args: argparse.Namespace,
) -> nn.Module:
    parts = []
    for speaker_id in training_speakers:
        indices = splits[speaker_id].train
        data.calculate_mu_sigma(indices)
        parts.append(_normalized_features(data, indices))
    return _fit_autoencoder(torch.cat(parts).to(device), args)


def _fit_logistic_regression(
    sober_train: Tensor, intoxicated_train: Tensor, c: float,
) -> Callable[[Tensor], Tensor]:
    from sklearn.linear_model import LogisticRegression

    features = torch.cat((sober_train, intoxicated_train))
    labels = [0] * len(sober_train) + [1] * len(intoxicated_train)
    model = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs", max_iter=1000)
    model.fit(features.numpy(), labels)

    def score(inputs: Tensor) -> Tensor:
        return torch.from_numpy(model.decision_function(inputs.numpy()))

    return score


def _speaker_scores(
    data: ALCData, split: SpeakerSplit, args: argparse.Namespace,
    device: torch.device, shared_model: nn.Module | None,
) -> SpeakerScores:
    train, calibration, test = _speaker_features(data, split.train, split.calibration, split.test, device)
    if args.method == "logistic":
        intoxicated_train = _normalized_features(data, split.intoxicated_enrollment)
        score = _fit_logistic_regression(train, intoxicated_train, args.lr_c)
        return SpeakerScores(score(calibration), score(test))
    if shared_model is not None:
        model = shared_model
    elif args.method == "pca":
        model = _fit_pca(train, args.pca_components)
    else:
        model = _fit_autoencoder(train, args)
    return SpeakerScores(
        _reconstruction_errors(model, calibration).cpu(),
        _reconstruction_errors(model, test).cpu(),
    )


def _speaker_metrics(
    splits: dict[int, SpeakerSplit], scores: dict[int, SpeakerScores], quantile: float | None,
) -> list[dict[str, int | float]]:
    rows = []
    for speaker_id, errors in scores.items():
        split = splits[speaker_id]
        rows.append({
            "speaker_id": speaker_id,
            "n_train_sober": len(split.train),
            "n_reserved_intoxicated": len(split.intoxicated_enrollment),
            "n_calibration_sober": len(split.calibration),
            "n_test_sober": len(split.sober_test),
            "n_test_intoxicated": len(split.intoxicated),
            **_score_errors(errors.calibration, errors.test, len(split.sober_test), quantile),
        })
    return rows


def _operating_metrics(rows: list[dict[str, int | float]]) -> dict[str, float]:
    return {
        **_aggregate_metrics(rows),
        "pooled_sober_fpr": sum(row["fp"] for row in rows) / sum(row["fp"] + row["tn"] for row in rows),
    }


def _threshold_sweep(
    splits: dict[int, SpeakerSplit], development_scores: dict[int, SpeakerScores], target_fpr: float,
) -> tuple[list[dict[str, float]], float | None]:
    rows = [
        {"quantile": quantile, **_operating_metrics(_speaker_metrics(splits, development_scores, quantile))}
        for quantile in SWEEP_QUANTILES
    ]
    feasible = [row for row in rows if row["pooled_sober_fpr"] <= target_fpr]
    if not feasible:
        return rows, None
    best = max(feasible, key=lambda row: (row["pooled_recall"], -row["pooled_sober_fpr"], row["quantile"]))
    return rows, best["quantile"]


def _macro_roc_recall_at_fpr(
    splits: dict[int, SpeakerSplit], scores: dict[int, SpeakerScores], target_fpr: float,
) -> float:
    import numpy as np
    from sklearn.metrics import roc_curve

    recalls = []
    for speaker_id, errors in scores.items():
        split = splits[speaker_id]
        labels = [0] * len(split.sober_test) + [1] * len(split.intoxicated)
        fpr, recall, _ = roc_curve(labels, errors.test.numpy())
        recalls.append(float(np.interp(target_fpr, fpr, recall)))
    return sum(recalls) / len(recalls)


def _diagnostic_rows(
    data: ALCData, splits: dict[int, SpeakerSplit],
    cohort_scores: dict[str, dict[int, SpeakerScores]], *, logistic: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    utterances = []
    pairs = []
    score_field = "decision_score" if logistic else "reconstruction_error"
    for cohort, scores in cohort_scores.items():
        for speaker_id, errors in scores.items():
            split = splits[speaker_id]
            metadata = {"cohort": cohort, "speaker_id": speaker_id}
            test_errors = dict(zip(split.test, errors.test.tolist(), strict=True))
            for partition, indices, values in (
                ("calibration", split.calibration, errors.calibration.tolist()),
                ("test", split.test, test_errors.values()),
            ):
                utterances.extend({
                    **metadata,
                    "split": partition,
                    "file": data.files[index],
                    "label": int(data.class_labels[index]),
                    score_field: error,
                } for index, error in zip(indices, values, strict=True))

            if logistic:
                continue
            sober_by_utterance_id = {data.files[index].split("_", 1)[0]: index for index in split.sober_test}
            for index in split.intoxicated:
                sober_index = sober_by_utterance_id.get(data.paired_utterance_ids[index])
                if sober_index is None:
                    continue
                sober_error = test_errors[sober_index]
                intoxicated_error = test_errors[index]
                pairs.append({
                    **metadata,
                    "prompt_id": data.files[index].split("_", 1)[0][-3:],
                    "sober_file": data.files[sober_index],
                    "intoxicated_file": data.files[index],
                    "sober_error": sober_error,
                    "intoxicated_error": intoxicated_error,
                    "error_difference": intoxicated_error - sober_error,
                })
    return utterances, pairs


def _csv_text(rows: Sequence[Mapping[str, object]]) -> str:
    with StringIO(newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()


def main() -> None:
    args = _parse_args()

    import mlflow
    from tqdm import tqdm

    from alc_data import ALCData

    logistic = args.method == "logistic"
    device = "cpu" if logistic else args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.manual_seed(args.seed)
    data = ALCData(seed=args.seed, feature_set=args.feature_set)
    reserve_intoxicated_training = logistic or args.reserve_intoxicated_training
    splits, skipped = _speaker_splits(
        data, args.seed, reserve_intoxicated_training=reserve_intoxicated_training,
    )

    mlflow.set_experiment("autoencoder")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            "dataset": "alc",
            "feature_set": data.cache_dict["feature_set"],
            "feature_level": data.cache_dict["feature_level"],
            "sober_split": "60/20/20",
            "intoxicated_split": "60/40" if reserve_intoxicated_training else "0/100",
            "method": args.method,
            "threshold_sweep": args.threshold_sweep,
            "seed": args.seed,
            "device": str(device),
        })
        if args.method == "pca":
            mlflow.log_param("pca_components", args.pca_components)
        elif logistic:
            mlflow.log_params({"lr_c": args.lr_c, "regularization": "L2", "class_weight": "balanced"})
        else:
            mlflow.log_params({
                "hidden_dimension": HIDDEN_DIMENSION,
                "bottleneck_dimension": BOTTLENECK_DIMENSION,
                "loss": "MSE",
                "optimizer": "Adam",
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
            })
        if data.files:
            mlflow.log_param("input_dimension", data.tensors_dict[data.files[0]].numel())
        mlflow.log_metric("speakers_skipped", len(skipped))
        if skipped:
            mlflow.log_dict({"speakers": skipped}, "skipped_speakers.json")
            print(f"Skipped {len(skipped)} speakers. See skipped_speakers.json for details.")
        if not splits:
            raise RuntimeError("No ALC speakers have enough sober and intoxicated data for evaluation")

        partitions = _speaker_partitions(list(splits), args.seed) if args.threshold_sweep else {"held_out": list(splits)}
        mlflow.log_dict(partitions, "speaker_partitions.json")
        shared_model = None
        if args.method == "shared-mlp":
            print(f"Training the shared autoencoder on {len(partitions['shared_training'])} speakers")
            shared_model = _fit_shared_autoencoder(data, splits, partitions["shared_training"], device, args)

        quantile = THRESHOLD_QUANTILE
        cohort_scores = {}
        for cohort in ("development", "held_out"):
            if cohort not in partitions:
                continue
            scores = {}
            for speaker_id in tqdm(partitions[cohort], desc=f"ALC {cohort} speakers"):
                errors = _speaker_scores(data, splits[speaker_id], args, device, shared_model)
                if not (torch.isfinite(errors.calibration).all() and torch.isfinite(errors.test).all()):
                    raise RuntimeError(f"Non-finite scores for speaker {speaker_id}")
                scores[speaker_id] = errors
            cohort_scores[cohort] = scores
            if cohort == "development":
                sweep, quantile = _threshold_sweep(splits, scores, args.target_fpr)
                mlflow.log_param("target_sober_fpr", args.target_fpr)
                mlflow.log_metric("speakers_development", len(scores))
                mlflow.log_text(_csv_text(sweep), "threshold_sweep.csv")
                if quantile is None:
                    print("No development cutoff met the false-positive limit. Using the always-sober rule.")

        mlflow.log_param("threshold_quantile", quantile if quantile is not None else "always_sober")
        rows = _speaker_metrics(splits, cohort_scores["held_out"], quantile)
        metrics = _operating_metrics(rows)
        if args.threshold_sweep:
            metrics["macro_roc_recall_at_target_fpr"] = _macro_roc_recall_at_fpr(
                splits, cohort_scores["held_out"], args.target_fpr,
            )
        utterances, pairs = _diagnostic_rows(data, splits, cohort_scores, logistic=logistic)
        metrics["speakers_evaluated"] = len(rows)
        if not logistic:
            held_out_pairs = [pair for pair in pairs if pair["cohort"] == "held_out"]
            metrics["matched_prompt_pairs"] = len(held_out_pairs)
            if held_out_pairs:
                higher = sum(pair["intoxicated_error"] > pair["sober_error"] for pair in held_out_pairs)
                metrics["matched_prompt_intoxicated_higher_fraction"] = higher / len(held_out_pairs)
        mlflow.log_metrics(metrics)
        mlflow.log_text(_csv_text(rows), "speaker_metrics.csv")
        mlflow.log_text(_csv_text(utterances), "utterance_scores.csv")
        if pairs:
            mlflow.log_text(_csv_text(pairs), "matched_prompts.csv")
        print(f"Evaluated {len(rows)} speakers in MLflow run {run.info.run_id}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
