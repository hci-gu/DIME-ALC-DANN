"""Save five participant-level OpenSMILE t-SNE plots for ALC, SE-DAC218 and GU."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import opensmile
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {"na": ("Sober", "o"), "a": ("Intoxicated", "^")}
DATASET_COLORS = {"ALC": "#0072B2", "SE-DAC218": "#D55E00", "GU": "#009E73"}
CONDITION_COLORS = {"na": "#0072B2", "a": "#D55E00"}

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


def participant_features(
    dataset: str,
    root: Path,
    cache_path: Path | None,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average each participant's utterance features within each labeled condition."""
    tensors = load_cache(cache_path)
    groups: dict[Participant, list[str]] = defaultdict(list)

    if dataset == "ALC":
        audio_dir = root / "wav" / "h"
        names = sorted(tensors) if tensors is not None else sorted(
            path.name for path in audio_dir.glob("*.wav")
        )
        labels_dir = root / "labels" / "h"
        for name in names:
            participant = alc_participant(name, labels_dir)
            if participant[1] != "cna":  # Match the ALC training cohort.
                groups[participant].append(name)
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

    participants = sorted(groups)
    if {condition for _, condition in participants} != CONDITIONS.keys():
        raise ValueError(f"{dataset} must contain both sober and intoxicated participant groups")

    if max_samples is not None and len(participants) > max_samples:
        rng = np.random.default_rng(seed)
        shuffled = [participants[index] for index in rng.permutation(len(participants))]
        # Keep at least one participant in each condition when subsampling.
        selected = {next(group for group in shuffled if group[1] == condition)
                    for condition in CONDITIONS}
        remaining = [group for group in shuffled if group not in selected]
        participants = sorted(selected.union(remaining[:max_samples - len(selected)]))

    if tensors is None:
        print(f"{dataset}: extracting OpenSMILE features from {audio_dir}")
        processor = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    features = []
    for participant in tqdm(participants, desc=f"Preparing {dataset} participants"):
        names = groups[participant]
        if tensors is None:
            utterances = np.concatenate([
                processor.process_file(str(audio_dir / name)).to_numpy()
                for name in names
            ])
        else:
            utterances = torch.stack([tensors[name] for name in names]).numpy()
        if utterances.ndim != 2 or len(utterances) != len(names) or not np.isfinite(utterances).all():
            raise ValueError(f"Invalid OpenSMILE features for {dataset} participant {participant[0]}")
        features.append(utterances.mean(axis=0, dtype=np.float64))
    return np.stack(features), np.array([condition for _, condition in participants])


def embed(features: np.ndarray, perplexity: float, seed: int, name: str) -> np.ndarray:
    """Standardize, reduce with PCA, and fit t-SNE without using class labels."""
    # Keep the neighborhood size small enough for datasets with few participants.
    perplexity = min(perplexity, max(1.0, (len(features) - 1) / 3))
    print(f"Fitting {name}: {len(features)} participant-condition points, perplexity {perplexity:g}")
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
) -> None:
    """Save a scatter plot with a labeled series for each group."""
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for points, label, color, marker in groups:
        ax.scatter(*points.T, s=30, alpha=0.7, color=color, marker=marker,
                   linewidths=0, label=f"{label} (n={len(points)})")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title=title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.supxlabel("Each point represents one participant in one condition", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset_names = {"alc": "ALC", "dac": "SE-DAC218", "gu": "GU"}
    for key, name in dataset_names.items():
        parser.add_argument(f"--{key}-dir", type=Path, default=ROOT / "data" / name)
        parser.add_argument(f"--{key}-cache", type=Path, help=f"Override the {name} feature cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--output", type=Path, help="Override the path of the combined dataset plot")
    parser.add_argument("--max-samples", type=int, help="Maximum participant-condition points per dataset")
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
        datasets[name] = participant_features(name, root, cache, args.max_samples, args.seed)
    all_features = np.concatenate([features for features, _ in datasets.values()])
    combined = embed(all_features, args.perplexity, args.seed, "all datasets")

    by_dataset = []
    by_dataset_and_condition = []
    offset = 0
    for name, (features, conditions) in datasets.items():
        points = combined[offset:offset + len(features)]
        offset += len(features)
        color = DATASET_COLORS[name]
        by_dataset.append((points, name, color, "o"))
        separate = embed(features, args.perplexity, args.seed, name)
        by_condition = []
        for condition, (label, marker) in CONDITIONS.items():
            mask = conditions == condition
            by_condition.append((separate[mask], label, CONDITION_COLORS[condition], marker))
            by_dataset_and_condition.append((points[mask], f"{name} {label.lower()}", color, marker))
        save_plot(args.output_dir / f"tsne-{name.lower()}-by-condition.png",
                  f"{name} OpenSMILE t-SNE by condition", by_condition)

    save_plot(args.output or args.output_dir / "tsne-all-by-dataset.png",
              "OpenSMILE t-SNE by dataset", by_dataset)
    save_plot(args.output_dir / "tsne-all-by-dataset-and-condition.png",
              "OpenSMILE t-SNE by dataset and condition", by_dataset_and_condition)


if __name__ == "__main__":
    main()
