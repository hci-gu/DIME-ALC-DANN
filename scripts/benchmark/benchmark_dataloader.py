import csv
import sys
import time
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alc_data import ALCData
from params import Params
from torch.utils.data import DataLoader, Subset


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_bool_list(value: str) -> list[bool]:
    bools = []
    for item in value.split(","):
        item = item.strip().lower()
        if item in {"true", "1", "yes", "y"}:
            bools.append(True)
        elif item in {"false", "0", "no", "n"}:
            bools.append(False)
        else:
            raise argparse.ArgumentTypeError(f"Invalid boolean value: {item}")
    return bools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ALCData batch loading speed.")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=parse_int_list, default=parse_int_list("0,1,2,4,8,12,16"))
    parser.add_argument("--pin-memory", type=parse_bool_list, default=parse_bool_list("false,true"))
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1999)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-transfer", action="store_true")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--plot-path", type=Path, default=Path(__file__).with_name("dataloader_benchmark.png"))
    return parser.parse_args()


def move_batch_to_device(batch, device: torch.device):
    x, y, s = batch
    return (
        x.to(device, non_blocking=True),
        y.to(device, dtype=torch.float32, non_blocking=True),
        s.to(device, dtype=torch.long, non_blocking=True),
    )


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_next_batch(
    loader_iter,
    loader: DataLoader,
    device: torch.device,
    include_transfer: bool,
) -> tuple[float, object, object]:
    start = time.perf_counter()
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)

    if include_transfer:
        batch = move_batch_to_device(batch, device)
        synchronize_if_needed(device)

    elapsed = time.perf_counter() - start
    return elapsed, loader_iter, batch


def benchmark_config(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    warmup_steps: int,
    sample_steps: int,
    device: torch.device,
    include_transfer: bool,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    loader_iter = iter(loader)

    for _ in range(warmup_steps):
        _, loader_iter, _ = time_next_batch(loader_iter, loader, device, include_transfer)

    timings = []
    for _ in range(sample_steps):
        elapsed, loader_iter, _ = time_next_batch(loader_iter, loader, device, include_transfer)
        timings.append(elapsed)

    timings_tensor = torch.tensor(timings)
    mean = timings_tensor.mean().item()
    std = timings_tensor.std(unbiased=True).item() if len(timings) > 1 else 0.0

    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "warmup_steps": warmup_steps,
        "sample_steps": sample_steps,
        "include_transfer": include_transfer,
        "device": str(device),
        "mean_s": mean,
        "std_s": std,
        "mean_ms": mean * 1000.0,
        "std_ms": std * 1000.0,
    }


def print_results(results: list[dict]) -> None:
    print()
    print("Batch timing results")
    print("-" * 88)
    print(
        f"{'workers':>8} {'pin':>6} {'transfer':>9} {'mean ms':>12} "
        f"{'std ms':>12} {'samples':>8} {'batch':>8}"
    )
    print("-" * 88)
    for result in results:
        print(
            f"{result['num_workers']:>8} "
            f"{str(result['pin_memory']):>6} "
            f"{str(result['include_transfer']):>9} "
            f"{result['mean_ms']:>12.2f} "
            f"{result['std_ms']:>12.2f} "
            f"{result['sample_steps']:>8} "
            f"{result['batch_size']:>8}"
        )


def write_csv(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def plot_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for pin_memory in sorted({result["pin_memory"] for result in results}):
        pin_results = sorted(
            [result for result in results if result["pin_memory"] == pin_memory],
            key=lambda result: result["num_workers"],
        )
        x = [result["num_workers"] for result in pin_results]
        y = [result["mean_ms"] for result in pin_results]
        yerr = [result["std_ms"] for result in pin_results]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=f"pin_memory={pin_memory}",
        )

    ax.set_title("Batch Loading Time vs num_workers")
    ax.set_xlabel("num_workers")
    ax.set_ylabel("Batch time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    p = Params.from_optional_overrides(
        dev_run=True,
        batch_size=args.batch_size,
        device=args.device,
    )

    device = torch.device(p.device)
    print(f"Using device: {device}")
    data = ALCData(max_samples=args.max_samples, seed=args.seed, verbose=True)
    train_indices, _, _ = data.speaker_split(
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
    )
    data.cache(train_indices)
    train_data = Subset(data, train_indices)

    results = []
    for num_workers in args.num_workers:
        for pin_memory in args.pin_memory:
            result = benchmark_config(
                dataset=train_data,
                batch_size=p.batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                warmup_steps=args.warmup_steps,
                sample_steps=args.sample_steps,
                device=device,
                include_transfer=args.include_transfer,
            )
            results.append(result)
            print(
                f"workers={num_workers}, pin_memory={pin_memory}: "
                f"{result['mean_ms']:.2f} +/- {result['std_ms']:.2f} ms"
            )

    print_results(results)
    if args.csv_path is not None:
        write_csv(args.csv_path, results)
        print(f"\nWrote CSV results to: {args.csv_path}")

    if args.plot_path is not None:
        plot_results(args.plot_path, results)
        print(f"Wrote plot to: {args.plot_path}")


if __name__ == "__main__":
    main()
