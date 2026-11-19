"""Save eight OpenSMILE t-SNE plots for ALC, SE-DAC218 and GU."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import opensmile
import torch
from matplotlib.colors import to_hex
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {"na": ("Sober", "o"), "a": ("Intoxicated", "^")}
DATASET_COLORS = {"ALC": "#0072B2", "SE-DAC218": "#D55E00", "GU": "#009E73"}
CONDITION_COLORS = {"na": "#0072B2", "a": "#D55E00"}
SPEAKER_MARKERS = ("o", "^", "s", "D", "P", "X", "v", "p", "<", ">", "h", "*")

type Participant = tuple[str, str]


def alc_participant(name: str, labels_dir: Path) -> Participant:
    """Read the speaker number and condition from the matching ALC annotation."""
    path = labels_dir / f"{Path(name).stem}_annot.json"
    with path.open(encoding="utf-8") as file:
        annotation = json.load(file)
    labels = annotation["levels"][0]["items"][0]["labels"]
    fields = {item["name"]: item["value"] for item in labels}
    return str(int(fields["spn"])), fields["alc"]


def load_cache(path: Path | None) -> dict[str, torch.Tensor] | None:
    """Load a ComParE functionals cache when available."""
    if path is None or not path.exists():
        return None
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if (cache.get("feature_set"), cache.get("feature_level")) != (
        "ComParE_2016", "Functionals"
    ):
        raise ValueError(f"Unexpected OpenSMILE feature configuration in {path}")
    return cache["tensors"]


def load_utterances(
    dataset: str,
    root: Path,
    cache_path: Path | None,
) -> tuple[np.ndarray, dict[Participant, slice]]:
    """Load utterance features with row ranges for each speaker and condition."""
    tensors = load_cache(cache_path)
    groups: dict[Participant, list[str]] = defaultdict(list)

    if dataset == "ALC":
        audio_dir = root / "wav" / "h"
        names = sorted(tensors) if tensors is not None else sorted(
            path.name for path in audio_dir.glob("*.wav")
        )
        labels_dir = root / "labels" / "h"
        for name in names:
            groups[alc_participant(name, labels_dir)].append(name)
    elif dataset == "SE-DAC218":
        audio_dir = root / "audio" / "wav"
        names = sorted(tensors) if tensors is not None else sorted(
            path.name for path in audio_dir.glob("*.wav")
        )
        with (root / "labels" / "labels.json").open(encoding="utf-8") as file:
            labels = {record["id"]: record["label"] for record in json.load(file)}
        for name in names:
            label = labels[Path(name).stem[:-8]]
            groups[(label["speaker"], label["label"])].append(name)
    else:
        audio_dir = root
        for folder, condition in (("filtered-control", "na"), ("filtered-intoxicated", "a")):
            directory = audio_dir / folder
            if not directory.is_dir():
                raise FileNotFoundError(f"GU directory not found: {directory}")
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() == ".wav":
                    speaker = str(int(path.stem.split("_")[-2]))
                    groups[(speaker, condition)].append(path.relative_to(audio_dir).as_posix())

    if not groups:
        raise ValueError(f"No utterances found for {dataset}")
    if tensors is None:
        print(f"{dataset}: extracting OpenSMILE features from {audio_dir}")
        processor = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    utterance_count = sum(len(names) for names in groups.values())
    features = None
    group_rows = {}
    offset = 0
    for participant, names in tqdm(sorted(groups.items()), desc=f"Preparing {dataset} utterances"):
        if tensors is None:
            utterances = np.concatenate([
                processor.process_file(str(audio_dir / name)).to_numpy()
                for name in names
            ])
        else:
            utterances = torch.stack([tensors[name] for name in names]).numpy()
        if utterances.ndim != 2 or len(utterances) != len(names) or not np.isfinite(utterances).all():
            raise ValueError(f"Invalid OpenSMILE features for {dataset} speaker {participant[0]}")
        if features is None:
            features = np.empty((utterance_count, utterances.shape[1]), dtype=np.float32)
        if utterances.shape[1] != features.shape[1]:
            raise ValueError(f"Inconsistent OpenSMILE feature dimensions for {dataset}")
        end = offset + len(utterances)
        features[offset:end] = utterances
        group_rows[participant] = slice(offset, end)
        offset = end
    assert features is not None
    return features, group_rows


def condition_features(
    dataset: str,
    utterances: np.ndarray,
    group_rows: dict[Participant, slice],
    max_samples: int | None,
    seed: int,
    *,
    aggregate_speakers: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare the samples for the five dataset and condition plots."""
    rows_by_condition = [
        (rows, condition) for (_, condition), rows in group_rows.items()
        if not (dataset == "ALC" and condition == "cna")  # Match the ALC training cohort.
    ]
    if {condition for _, condition in rows_by_condition} != CONDITIONS.keys():
        raise ValueError(f"{dataset} must contain both sober and intoxicated samples")

    if aggregate_speakers:
        conditions = np.array([condition for _, condition in rows_by_condition])
    else:
        conditions = np.repeat(
            [condition for _, condition in rows_by_condition],
            [rows.stop - rows.start for rows, _ in rows_by_condition],
        )
        if len(conditions) == len(utterances) and (
            max_samples is None or max_samples >= len(utterances)
        ):
            return utterances, conditions

    selected = np.arange(len(conditions))
    if max_samples is not None and len(selected) > max_samples:
        shuffled = np.random.default_rng(seed).permutation(selected)
        # Keep at least one sample in each condition when subsampling.
        required = [shuffled[conditions[shuffled] == condition][0] for condition in CONDITIONS]
        remaining = shuffled[~np.isin(shuffled, required)]
        selected = np.sort(np.concatenate((required, remaining[:max_samples - len(required)])))

    if aggregate_speakers:
        features = np.stack([
            utterances[rows_by_condition[index][0]].mean(axis=0, dtype=np.float64)
            for index in selected
        ])
    else:
        eligible_rows = np.concatenate([np.arange(rows.start, rows.stop) for rows, _ in rows_by_condition])
        features = utterances[eligible_rows[selected]]
    return features, conditions[selected]


def embed(features: np.ndarray, perplexity: float, seed: int, name: str) -> np.ndarray:
    """Standardize, reduce with PCA, and fit t-SNE without using class labels."""
    # Keep the neighborhood size small enough for datasets with few points.
    perplexity = min(perplexity, max(1.0, (len(features) - 1) / 3))
    print(f"Fitting {name}: {len(features)} points, perplexity {perplexity:g}")
    scaled = StandardScaler().fit_transform(features)
    n_components = min(100, scaled.shape[0] - 1, scaled.shape[1])
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(scaled)
    return TSNE(
        n_components=2, perplexity=perplexity, random_state=seed, init="random"
    ).fit_transform(reduced)


def save_plot(
    path: Path,
    title: str,
    groups: list[tuple[np.ndarray, str, str, str]],
    point_description: str,
    *,
    show_legend: bool = True,
) -> None:
    """Save a scatter plot with a labeled series for each group."""
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for points, label, color, marker in groups:
        ax.scatter(*points.T, s=30 if show_legend else 12, alpha=0.7, color=color, marker=marker,
                   linewidths=0, label=f"{label} (n={len(points)})")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title=title)
    ax.set_aspect("equal", adjustable="datalim")
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.supxlabel(f"Each point represents {point_description}", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def save_speaker_plot(
    path: Path, dataset: str, embedding: np.ndarray, group_rows: dict[Participant, slice],
) -> None:
    """Style each speaker's utterance points consistently across conditions."""
    speaker_points: dict[str, list[np.ndarray]] = defaultdict(list)
    for (speaker, condition), rows in group_rows.items():
        # GU's source speaker IDs are local to each cohort.
        speaker_id = f"{condition}/{speaker}" if dataset == "GU" else speaker
        speaker_points[speaker_id].append(embedding[rows])

    palette = [
        to_hex(color)
        for name in ("tab20", "tab20b", "tab20c")
        for color in plt.get_cmap(name).colors
    ]
    groups = []
    for index, (speaker, points) in enumerate(speaker_points.items()):
        cycle, color_index = divmod(index, len(palette))
        marker = SPEAKER_MARKERS[(cycle + color_index) % len(SPEAKER_MARKERS)]
        groups.append((np.concatenate(points), speaker, palette[color_index], marker))

    save_plot(path, f"{dataset} OpenSMILE t-SNE by speaker ({len(groups)} speakers)",
              groups, "one utterance", show_legend=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset_names = {"alc": "ALC", "dac": "SE-DAC218", "gu": "GU"}
    for key, name in dataset_names.items():
        parser.add_argument(f"--{key}-dir", type=Path, default=ROOT / "data" / name)
        parser.add_argument(f"--{key}-cache", type=Path, help=f"Override the {name} feature cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--output", type=Path, help="Override the path of the combined dataset plot")
    parser.add_argument(
        "--aggregate-speakers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average utterances in dataset and condition plots (default: enabled)",
    )
    parser.add_argument(
        "--max-samples", type=int,
        help="Maximum points per dataset in dataset and condition plots",
    )
    parser.add_argument("--perplexity", type=float, default=30, help="Maximum t-SNE perplexity (default: 30)")
    parser.add_argument("--seed", type=int, default=1999)
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples < 2:
        parser.error("--max-samples must be at least 2 to include both conditions")
    if not np.isfinite(args.perplexity) or args.perplexity <= 0:
        parser.error("--perplexity must be a positive finite number")

    datasets = {}
    for key, name in dataset_names.items():
        root = getattr(args, f"{key}_dir")
        cache = getattr(args, f"{key}_cache")
        if cache is None and root.resolve() == ROOT / "data" / name:
            cache = ROOT / ".cache" / f"{key}-opensmile-features.pt"
        utterances, group_rows = load_utterances(name, root, cache)
        features, conditions = condition_features(
            name, utterances, group_rows, args.max_samples, args.seed,
            aggregate_speakers=args.aggregate_speakers,
        )
        speaker_embedding = embed(utterances, args.perplexity, args.seed, f"{name} by speaker")
        separate = speaker_embedding if features is utterances else embed(
            features, args.perplexity, args.seed, f"{name} by condition"
        )
        save_speaker_plot(
            args.output_dir / f"tsne-{name.lower()}-by-speaker.png", name, speaker_embedding, group_rows,
        )
        datasets[name] = (features, conditions, separate)
        del utterances

    all_features = np.concatenate([features for features, _, _ in datasets.values()])
    combined = embed(all_features, args.perplexity, args.seed, "all datasets")
    point_description = (
        "one participant in one condition" if args.aggregate_speakers else "one utterance"
    )

    by_dataset = []
    by_dataset_and_condition = []
    offset = 0
    for name, (features, conditions, separate) in datasets.items():
        points = combined[offset:offset + len(features)]
        offset += len(features)
        color = DATASET_COLORS[name]
        by_dataset.append((points, name, color, "o"))
        by_condition = []
        for condition, (label, marker) in CONDITIONS.items():
            mask = conditions == condition
            by_condition.append((separate[mask], label, CONDITION_COLORS[condition], marker))
            by_dataset_and_condition.append((points[mask], f"{name} {label.lower()}", color, marker))
        save_plot(args.output_dir / f"tsne-{name.lower()}-by-condition.png",
                  f"{name} OpenSMILE t-SNE by condition", by_condition, point_description)

    save_plot(args.output or args.output_dir / "tsne-all-by-dataset.png",
              "OpenSMILE t-SNE by dataset", by_dataset, point_description)
    save_plot(args.output_dir / "tsne-all-by-dataset-and-condition.png",
              "OpenSMILE t-SNE by dataset and condition", by_dataset_and_condition,
              point_description)


if __name__ == "__main__":
    main()
