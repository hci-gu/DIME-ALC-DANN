import csv
import json

from datetime import datetime
from pathlib import Path

import mlflow
import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from alc_data import ALCData
from dac218_data import DAC218Data
from model import DANN
from utils.argument_parsing import parse_args


EXPERIMENT_NAME = "DANN"
SPLIT_ARTIFACT = "speaker_data_split.json"
BAC_BIN_WIDTH = 0.2


class ClassifierInference(nn.Module):
    """Classifier-only DANN path used for eager and compiled inference."""

    def __init__(self, model: DANN):
        super().__init__()
        self.extractor = model.extractor
        self.classifier = model.classifier

    def forward(self, x: Tensor) -> Tensor:
        features = self.extractor(x.squeeze(1))
        return self.classifier(features)


class IndexedSubset(Dataset):
    """Subset that preserves each sample's index in the underlying dataset."""

    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        dataset_index = self.indices[index]
        x, y, _ = self.dataset[dataset_index]
        return x, y, dataset_index


def resolve_run(client: MlflowClient, run_name: str) -> Run:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Could not find MLflow experiment: {EXPERIMENT_NAME}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=50_000,
    )
    matching_runs = [
        run
        for run in runs
        if run.data.tags.get("mlflow.runName") == run_name
    ]

    if not matching_runs:
        raise RuntimeError(
            f"Could not find a run named {run_name!r} in experiment {EXPERIMENT_NAME!r}"
        )
    if len(matching_runs) > 1:
        run_ids = [run.info.run_id for run in matching_runs]
        raise RuntimeError(
            f"Run name {run_name!r} is not unique. Matching run IDs: {run_ids}"
        )

    run = matching_runs[0]
    if run.info.status != "FINISHED":
        raise RuntimeError(
            f"Run {run_name!r} has status {run.info.status!r}; expected 'FINISHED'"
        )
    if run.data.tags.get("run_type") != "normal":
        raise RuntimeError(
            f"Run {run_name!r} has run_type={run.data.tags.get('run_type')!r}; "
            "only normal DANN runs are supported"
        )
    return run


def resolve_max_samples(params: dict[str, str]) -> int | None:
    logged_value = params.get("max_samples")
    if logged_value is not None:
        if logged_value.lower() in {"all", "none"}:
            return None
        try:
            max_samples = int(logged_value)
        except ValueError as error:
            raise RuntimeError(
                f"Invalid logged max_samples value: {logged_value!r}"
            ) from error
        if max_samples <= 0:
            raise RuntimeError(
                f"Logged max_samples must be positive or 'all', got {logged_value!r}"
            )
        return max_samples

    dev_run = params.get("dev_run", "false").lower() == "true"
    inferred_value = 1000 if dev_run else None
    inferred_description = str(inferred_value) if inferred_value is not None else "all"
    print(
        "Warning: this run predates max_samples logging. "
        f"Inferring max_samples={inferred_description} from dev_run; an explicit "
        "historical --max-samples value cannot be recovered."
    )
    return inferred_value


def create_dataset(
    dataset_name: str,
    seed: int,
    max_samples: int | None,
    verbose: bool,
):
    dataset_classes = {
        "alc": ALCData,
        "dac": DAC218Data,
    }
    dataset_class = dataset_classes.get(dataset_name)
    if dataset_class is None:
        raise RuntimeError(
            f"Unsupported dataset {dataset_name!r}; expected one of {sorted(dataset_classes)}"
        )
    return dataset_class(max_samples=max_samples, seed=seed, verbose=verbose)


def load_speaker_split(path: str) -> dict[str, set[int]]:
    with open(path, "r", encoding="utf-8") as split_file:
        raw_split = json.load(split_file)

    required_keys = {"train_speakers", "val_speakers", "test_speakers"}
    missing_keys = required_keys - set(raw_split)
    if missing_keys:
        raise RuntimeError(f"Split artifact is missing keys: {sorted(missing_keys)}")

    split: dict[str, set[int]] = {}
    for key in sorted(required_keys):
        values = raw_split[key]
        if not isinstance(values, list) or not values:
            raise RuntimeError(f"Split artifact field {key!r} must be a non-empty list")
        try:
            speakers = [int(value) for value in values]
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Split artifact field {key!r} contains a non-integer speaker ID"
            ) from error
        if len(speakers) != len(set(speakers)):
            raise RuntimeError(f"Split artifact field {key!r} contains duplicates")
        split[key] = set(speakers)

    if split["train_speakers"] & split["val_speakers"]:
        raise RuntimeError("Training and validation speaker sets overlap")
    if split["train_speakers"] & split["test_speakers"]:
        raise RuntimeError("Training and test speaker sets overlap")
    if split["val_speakers"] & split["test_speakers"]:
        raise RuntimeError("Validation and test speaker sets overlap")
    return split


def apply_speaker_split(
    dataset,
    split: dict[str, set[int]],
) -> tuple[list[int], list[int], list[int]]:
    available_speakers = set(dataset.speaker_ids)
    logged_speakers = set().union(*split.values())
    missing_speakers = logged_speakers - available_speakers
    if missing_speakers:
        raise RuntimeError(
            "The current dataset is missing speakers from the logged split: "
            f"{sorted(missing_speakers)}"
        )
    extra_speakers = available_speakers - logged_speakers
    if extra_speakers:
        print(
            "Warning: the current dataset contains speakers that are not present "
            "in the logged split; their samples will be ignored."
        )

    dataset.train_speakers_id = split["train_speakers"]
    dataset.val_speakers_id = split["val_speakers"]
    dataset.test_speakers_id = split["test_speakers"]
    dataset.train_speaker_mapping = {
        speaker_id: local_index
        for local_index, speaker_id in enumerate(sorted(dataset.train_speakers_id))
    }
    dataset.is_split = True

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for index, speaker_id in enumerate(dataset.speaker_ids):
        if speaker_id in dataset.train_speakers_id:
            train_indices.append(index)
        elif speaker_id in dataset.val_speakers_id:
            val_indices.append(index)
        elif speaker_id in dataset.test_speakers_id:
            test_indices.append(index)

    for split_name, indices in (
        ("training", train_indices),
        ("validation", val_indices),
        ("test", test_indices),
    ):
        if not indices:
            raise RuntimeError(f"The reconstructed {split_name} split is empty")
    return train_indices, val_indices, test_indices


@torch.inference_mode()
def predict(
    predictor: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for x, y, index in loader:
        logits = predictor(x.to(device)).squeeze(-1)
        dataset_indices.append(index.numpy())
        labels.append(y.numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())

    return (
        np.concatenate(dataset_indices),
        np.concatenate(labels).astype(bool),
        np.concatenate(probabilities),
    )


def select_validation_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(labels)) != 2:
        raise RuntimeError(
            "The validation split must contain both sober and intoxicated samples "
            "to select an F1 threshold"
        )

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    return float(thresholds[f1_scores.argmax()])


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, int | float | None], np.ndarray]:
    predictions = probabilities >= threshold
    tp = int((predictions & labels).sum())
    tn = int((~predictions & ~labels).sum())
    fp = int((predictions & ~labels).sum())
    fn = int((~predictions & labels).sum())

    def safe_divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    has_both_classes = len(np.unique(labels)) == 2
    metrics: dict[str, int | float | None] = {
        "count": len(labels),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "auroc": float(roc_auc_score(labels, probabilities)) if has_both_classes else None,
        "threshold": threshold,
    }
    return metrics, predictions


def write_predictions(
    dataset,
    dataset_indices: np.ndarray,
    bac_values: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    run_name: str,
    run_id: str,
    output_path: Path,
) -> Path:
    fieldnames = [
        "dataset_index",
        "filename",
        "speaker_id",
        "bac_per_mille",
        "true_label",
        "probability_intoxicated",
        "predicted_label",
        "threshold",
        "run_name",
        "run_id",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, bac, label, probability, prediction in zip(
            dataset_indices,
            bac_values,
            labels,
            probabilities,
            predictions,
        ):
            dataset_index = int(index)
            writer.writerow(
                {
                    "dataset_index": dataset_index,
                    "filename": dataset.files[dataset_index],
                    "speaker_id": dataset.speaker_ids[dataset_index],
                    "bac_per_mille": float(bac),
                    "true_label": int(label),
                    "probability_intoxicated": float(probability),
                    "predicted_label": int(prediction),
                    "threshold": threshold,
                    "run_name": run_name,
                    "run_id": run_id,
                }
            )
    return output_path.resolve()


def create_output_paths(dataset_name: str, run_name: str) -> dict[str, Path]:
    output_directory = Path("evals")
    output_directory.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"{dataset_name}-{sanitize_run_name(run_name)}-{timestamp}"
    return {
        "predictions": output_directory / f"predictions-{suffix}.csv",
        "bac_bins": output_directory / f"bac-bins-{suffix}.csv",
        "bac_probability": output_directory / f"bac-probability-{suffix}.png",
        "bac_precision_recall": output_directory / f"bac-precision-recall-{suffix}.png",
        "bac_pod": output_directory / f"bac-pod-{suffix}.png",
    }


def wilson_interval(successes: int, total: int, z_score: float = 1.96) -> tuple[float, float] | None:
    if total == 0:
        return None
    proportion = successes / total
    z_squared = z_score ** 2
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    half_width = (
        z_score
        * np.sqrt(
            proportion * (1 - proportion) / total
            + z_squared / (4 * total ** 2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def calculate_bac_bins(
    bac_values: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    bin_width: float = BAC_BIN_WIDTH,
) -> list[dict[str, int | float | None]]:
    if len(bac_values) != len(labels) or len(labels) != len(predictions):
        raise ValueError("BAC values, labels, and predictions must have equal lengths")
    if not len(bac_values):
        raise ValueError("Cannot calculate BAC bins without samples")
    if bin_width <= 0:
        raise ValueError("BAC bin width must be positive")
    if not np.isfinite(bac_values).all() or (bac_values < 0).any():
        raise ValueError("BAC values must be finite and non-negative")

    n_bins = max(1, int(np.ceil(float(bac_values.max()) / bin_width)))
    edges = np.arange(n_bins + 1, dtype=float) * bin_width
    bin_rows: list[dict[str, int | float | None]] = []

    for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if bin_index == n_bins - 1:
            in_bin = (bac_values >= lower) & (bac_values <= upper)
        else:
            in_bin = (bac_values >= lower) & (bac_values < upper)

        bin_labels = labels[in_bin]
        bin_predictions = predictions[in_bin]
        tp = int((bin_predictions & bin_labels).sum())
        fp = int((bin_predictions & ~bin_labels).sum())
        fn = int((~bin_predictions & bin_labels).sum())
        n_intoxicated = int(bin_labels.sum())
        detected = tp
        missed = fn
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        interval = wilson_interval(detected, detected + missed)

        bin_rows.append(
            {
                "bac_lower": float(lower),
                "bac_upper": float(upper),
                "bac_midpoint": float((lower + upper) / 2),
                "n_samples": int(in_bin.sum()),
                "n_sober": int(len(bin_labels) - n_intoxicated),
                "n_intoxicated": n_intoxicated,
                "predicted_positive": int(bin_predictions.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "detected": detected,
                "missed": missed,
                "precision": precision,
                "recall": recall,
                "pod": recall,
                "pod_ci_lower": interval[0] if interval is not None else None,
                "pod_ci_upper": interval[1] if interval is not None else None,
            }
        )
    return bin_rows


def write_bac_bins(
    bin_rows: list[dict[str, int | float | None]],
    output_path: Path,
) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(bin_rows[0]))
        writer.writeheader()
        writer.writerows(bin_rows)
    return output_path.resolve()


def plot_bac_probability(
    bac_values: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(7, 5), dpi=140)
    for label, name, color in (
        (False, "Sober", "tab:blue"),
        (True, "Intoxicated", "tab:orange"),
    ):
        mask = labels == label
        axis.scatter(
            bac_values[mask],
            probabilities[mask],
            alpha=0.55,
            s=22,
            color=color,
            label=f"{name} (n={int(mask.sum())})",
        )
    axis.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Decision threshold={threshold:.3f}",
    )
    axis.set(
        title="BAC versus predicted intoxication probability",
        xlabel="BAC (‰)",
        ylabel="P(intoxicated)",
        ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path.resolve()


def plot_bac_precision_recall(
    bin_rows: list[dict[str, int | float | None]],
    output_path: Path,
) -> Path:
    midpoints = np.array([row["bac_midpoint"] for row in bin_rows], dtype=float)
    precision = np.array(
        [np.nan if row["precision"] is None else row["precision"] for row in bin_rows],
        dtype=float,
    )
    recall = np.array(
        [np.nan if row["recall"] is None else row["recall"] for row in bin_rows],
        dtype=float,
    )

    figure, axis = plt.subplots(figsize=(7, 5), dpi=140)
    axis.plot(midpoints, precision, marker="o", label="Precision")
    axis.plot(midpoints, recall, marker="o", label="Recall")
    for midpoint, row in zip(midpoints, bin_rows):
        axis.annotate(
            f"n={row['n_samples']}",
            (midpoint, 1.01),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )
    axis.set(
        title=f"Precision and recall by {BAC_BIN_WIDTH:.1f}‰ BAC bin",
        xlabel="BAC bin midpoint (‰)",
        ylabel="Metric value",
        ylim=(-0.02, 1.13),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path.resolve()


def plot_bac_pod(
    bin_rows: list[dict[str, int | float | None]],
    output_path: Path,
) -> Path:
    populated_rows = [row for row in bin_rows if row["n_intoxicated"] > 0]
    if not populated_rows:
        figure, axis = plt.subplots(figsize=(7, 5), dpi=140)
        axis.text(
            0.5,
            0.5,
            "No intoxicated clips in the selected test samples",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_title("Probability of detection by BAC")
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(output_path)
        plt.close(figure)
        return output_path.resolve()

    midpoints = np.array([row["bac_midpoint"] for row in populated_rows], dtype=float)
    detected = np.array([row["detected"] for row in populated_rows], dtype=int)
    missed = np.array([row["missed"] for row in populated_rows], dtype=int)
    pod = np.array([row["pod"] for row in populated_rows], dtype=float)
    ci_lower = np.array([row["pod_ci_lower"] for row in populated_rows], dtype=float)
    ci_upper = np.array([row["pod_ci_upper"] for row in populated_rows], dtype=float)

    figure, count_axis = plt.subplots(figsize=(7, 5), dpi=140)
    bar_width = BAC_BIN_WIDTH * 0.72
    count_axis.bar(midpoints, detected, width=bar_width, label="Detected", color="tab:green")
    count_axis.bar(
        midpoints,
        missed,
        width=bar_width,
        bottom=detected,
        label="Missed",
        color="tab:red",
    )
    count_axis.set(
        title="Probability of detection by BAC",
        xlabel="BAC bin midpoint (‰)",
        ylabel="Intoxicated clip count",
    )

    pod_axis = count_axis.twinx()
    pod_axis.errorbar(
        midpoints,
        pod,
        yerr=np.vstack((pod - ci_lower, ci_upper - pod)),
        color="black",
        marker="o",
        capsize=4,
        linewidth=1.5,
        label="POD (95% Wilson CI)",
    )
    pod_axis.set_ylabel("Probability of detection")
    pod_axis.set_ylim(-0.02, 1.05)
    count_axis.grid(axis="y", alpha=0.25)

    count_handles, count_labels = count_axis.get_legend_handles_labels()
    pod_handles, pod_labels = pod_axis.get_legend_handles_labels()
    count_axis.legend(count_handles + pod_handles, count_labels + pod_labels, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path.resolve()


def sanitize_run_name(run_name: str) -> str:
    return run_name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested_device)


def main() -> None:
    args = parse_args(profile="inference")
    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    run = resolve_run(client, args.run_name)
    params = run.data.params

    dataset_name = params.get("dataset")
    if dataset_name is None:
        raise RuntimeError(
            "The selected run does not log a dataset and cannot be resolved automatically"
        )
    checkpoint_path = Path("weights") / (
        f"dann_model-{dataset_name}-{sanitize_run_name(args.run_name)}.pth"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Could not find the checkpoint for this run: {checkpoint_path}"
        )
    try:
        seed = int(params["seed"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("The selected run does not contain a valid seed") from error

    max_samples = resolve_max_samples(params)
    dataset = create_dataset(dataset_name, seed, max_samples, args.verbose)

    try:
        split_path = client.download_artifacts(run.info.run_id, SPLIT_ARTIFACT)
    except Exception as error:
        raise RuntimeError(
            f"Could not download {SPLIT_ARTIFACT!r} for run {args.run_name!r}"
        ) from error
    split = load_speaker_split(split_path)
    train_indices, val_indices, test_indices = apply_speaker_split(dataset, split)
    dataset.cache(train_indices=train_indices)

    if args.max_test_samples is not None:
        test_indices = test_indices[:args.max_test_samples]
    if not test_indices:
        raise RuntimeError("No test samples were selected")

    batch_size = args.batch_size or int(params.get("batch_size", 128))
    n_workers = args.n_workers
    if n_workers is None:
        n_workers = int(params.get("n_workers", 1))
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    val_loader = DataLoader(
        IndexedSubset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        IndexedSubset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=pin_memory,
    )

    model = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(model, DANN):
        raise RuntimeError(
            f"Expected a DANN checkpoint, found {type(model).__name__}"
        )
    model.to(device).eval()
    predictor: nn.Module = ClassifierInference(model).to(device).eval()
    if args.compile:
        try:
            predictor = torch.compile(predictor)
        except Exception as error:
            raise RuntimeError("Could not compile the inference model") from error

    try:
        _, val_labels, val_probabilities = predict(predictor, val_loader, device)
        threshold = select_validation_threshold(val_labels, val_probabilities)
        test_dataset_indices, test_labels, test_probabilities = predict(
            predictor,
            test_loader,
            device,
        )
    except Exception as error:
        if args.compile:
            raise RuntimeError(
                "Compiled inference failed while executing the model"
            ) from error
        raise

    metrics, test_predictions = calculate_metrics(
        test_labels,
        test_probabilities,
        threshold,
    )
    if len(dataset.bac_values) != len(dataset.files):
        raise RuntimeError("Dataset BAC values are not aligned with its files")
    test_bac_values = np.array(
        [dataset.bac_values[int(index)] for index in test_dataset_indices],
        dtype=float,
    )
    bin_rows = calculate_bac_bins(
        test_bac_values,
        test_labels,
        test_predictions,
    )
    output_paths = create_output_paths(dataset_name, args.run_name)
    generated_paths = [write_predictions(
        dataset=dataset,
        dataset_indices=test_dataset_indices,
        bac_values=test_bac_values,
        labels=test_labels,
        probabilities=test_probabilities,
        predictions=test_predictions,
        threshold=threshold,
        run_name=args.run_name,
        run_id=run.info.run_id,
        output_path=output_paths["predictions"],
    )]
    generated_paths.append(write_bac_bins(bin_rows, output_paths["bac_bins"]))
    generated_paths.append(
        plot_bac_probability(
            test_bac_values,
            test_labels,
            test_probabilities,
            threshold,
            output_paths["bac_probability"],
        )
    )
    generated_paths.append(
        plot_bac_precision_recall(
            bin_rows,
            output_paths["bac_precision_recall"],
        )
    )
    generated_paths.append(plot_bac_pod(bin_rows, output_paths["bac_pod"]))

    print(f"Run: {args.run_name} ({run.info.run_id})")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}; compiled: {args.compile}")
    print(json.dumps(metrics, indent=2))
    print("Generated evaluation files:")
    for generated_path in generated_paths:
        print(f"- {generated_path}")


if __name__ == "__main__":
    main()
