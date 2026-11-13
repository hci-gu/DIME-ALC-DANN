"""Print audio durations and plot their distribution for a directory."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import soundfile as sf


AUDIO_EXTENSIONS = {
    ".aif", ".aiff", ".au", ".flac", ".mp3", ".ogg", ".opus",
    ".wav", ".w64", ".caf", ".wve", ".nist", ".voc", ".svx",
}


def audio_duration(path: Path) -> float:
    """Decode an audio file and return its actual duration in seconds."""
    with sf.SoundFile(path) as audio:
        frames = sum(len(block) for block in audio.blocks(blocksize=65536))
        return frames / audio.samplerate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing audio files")
    parser.add_argument("--recursive", action="store_true", help="Include subdirectories")
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins (default: 30)")
    parser.add_argument("--output", type=Path, help="Save the histogram to this file instead of displaying it")
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"Not a directory: {args.directory}")
    if args.bins < 1:
        parser.error("--bins must be a positive integer")

    paths = args.directory.rglob("*") if args.recursive else args.directory.iterdir()
    files = sorted(path for path in paths if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)
    if not files:
        parser.error(f"No supported audio files found in {args.directory}")

    durations = []
    failures = []
    for path in files:
        try:
            durations.append((audio_duration(path), path))
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append((path, exc))

    print("Audio lengths (shortest first):")
    for duration, path in sorted(durations):
        print(f"{duration:10.3f} s  {path.relative_to(args.directory)}")

    if failures:
        print("\nFiles that could not be read:")
        for path, exc in failures:
            print(f"  {path.relative_to(args.directory)}: {exc}")

    if not durations:
        raise SystemExit("No audio files could be read.")

    plt.hist([duration for duration, _ in durations], bins=args.bins, edgecolor="black")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of files")
    plt.title("Audio duration distribution")
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output)
        print(f"\nHistogram saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
