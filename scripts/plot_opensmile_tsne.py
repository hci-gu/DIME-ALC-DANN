"""Visualize cached ALC and DAC OpenSMILE features with t-SNE."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_features(path: Path, max_samples: int | None, seed: int) -> np.ndarray:
    """Load a reproducible sample of feature vectors from an OpenSMILE cache."""
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if (cache.get("feature_set"), cache.get("feature_level")) != (
        "ComParE_2016", "Functionals"
    ):
        raise ValueError(f"Unexpected OpenSMILE feature configuration in {path}")

    tensors = cache["tensors"]
    names = sorted(tensors)
    if max_samples is not None and len(names) > max_samples:
        indices = np.random.default_rng(seed).choice(len(names), max_samples, replace=False)
        names = [names[index] for index in sorted(indices)]
    if not names:
        raise ValueError(f"No features found in {path}")
    features = torch.stack([tensors[name] for name in names]).numpy()
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError(f"Expected a finite matrix of feature vectors in {path}")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alc-cache", type=Path, default=Path(".cache/alc-opensmile-features.pt"))
    parser.add_argument("--dac-cache", type=Path, default=Path(".cache/dac-opensmile-features.pt"))
    parser.add_argument("--max-samples", type=int, help="Maximum samples per dataset (default: all)")
    parser.add_argument("--perplexity", type=float, default=30, help="t-SNE perplexity (default: 30)")
    parser.add_argument("--seed", type=int, default=1999)
    parser.add_argument("--output", type=Path, help="Save the plot instead of displaying it")
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if not np.isfinite(args.perplexity) or args.perplexity <= 0:
        parser.error("--perplexity must be a positive finite number")

    alc = load_features(args.alc_cache, args.max_samples, args.seed)
    dac = load_features(args.dac_cache, args.max_samples, args.seed)
    if alc.shape[1] != dac.shape[1]:
        raise ValueError("ALC and DAC caches have different feature dimensions")
    features = np.concatenate((alc, dac))
    if args.perplexity >= len(features):
        parser.error("--perplexity must be smaller than the total number of samples")

    print(f"ALC: {len(alc)} samples; DAC: {len(dac)} samples; features: {features.shape[1]}")
    features = StandardScaler().fit_transform(features)
    n_components = min(100, features.shape[0] - 1, features.shape[1])
    features = PCA(n_components=n_components, random_state=args.seed).fit_transform(features)
    embedding = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed,
                     init="random").fit_transform(features)

    fig, ax = plt.subplots()
    ax.scatter(embedding[:len(alc), 0], embedding[:len(alc), 1], s=8, alpha=0.5, label="ALC")
    ax.scatter(embedding[len(alc):, 0], embedding[len(alc):, 1], s=8, alpha=0.5, label="DAC")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="OpenSMILE features: ALC and DAC")
    ax.legend(markerscale=2)
    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=200)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
