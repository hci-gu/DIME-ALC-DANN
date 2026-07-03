import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alc_data import ALCData
from model import DANN
from params import Params
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader, random_split
from utils.compute_params import alpha_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile DANN training with PyTorch profiler.")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "profiling" / "tensorboard")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1999)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wait", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--active", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--record-shapes", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_training_objects(args: argparse.Namespace):
    p = Params(dev_run=True)
    if args.batch_size is not None:
        p.batch_size = args.batch_size
    if args.num_workers is not None:
        p.n_workers = args.num_workers
    if args.device is not None:
        p.device = args.device

    device = torch.device(p.device)

    data = ALCData(max_samples=args.max_samples, verbose=True)
    generator = torch.Generator().manual_seed(args.seed)
    train_data, _, _ = random_split(data, [0.8, 0.1, 0.1], generator=generator)
    pos_weight = data.calculate_pos_weight(train_data.indices).to(device) if p.use_pos_weight else None
    p.discriminator_output_dimension = len(data.speaker_id_to_index)

    train_loader = DataLoader(
        train_data,
        batch_size=p.batch_size,
        shuffle=True,
        num_workers=p.n_workers,
        pin_memory=p.pin_memory,
    )

    model = DANN(p).to(device)
    optimizer_cls = getattr(torch.optim, p.optimizer)
    optimizer = optimizer_cls(model.parameters(), **p.get_vars_from_prefix("optimizer"))
    loss_functions = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        nn.CrossEntropyLoss(),
    )

    return model, p, optimizer, loss_functions, train_loader, device


def train_one_profiled_window(
    model: DANN,
    p: Params,
    optimizer: torch.optim.Optimizer,
    loss_functions: tuple[nn.BCEWithLogitsLoss, nn.CrossEntropyLoss],
    train_loader: DataLoader,
    device: torch.device,
    profiler,
    n_steps: int,
) -> None:
    classifier_loss_fn, discriminator_loss_fn = loss_functions
    train_iter = iter(train_loader)
    model.train()

    for step in range(n_steps):
        try:
            with record_function("dataloader_next"):
                x, y, s = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            with record_function("dataloader_next"):
                x, y, s = next(train_iter)

        alpha = alpha_schedule(step, n_steps)

        with record_function("device_transfer"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            s = s.to(device, dtype=torch.long, non_blocking=True)

        with record_function("forward"):
            class_logits, speaker_logits = model(x, alpha=alpha)
            classifier_loss = classifier_loss_fn(class_logits.squeeze(-1), y)
            discriminator_loss = discriminator_loss_fn(speaker_logits, s)
            loss = classifier_loss + discriminator_loss

        with record_function("backward"):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

        with record_function("optimizer_step"):
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        profiler.step()


def main() -> None:
    args = parse_args()
    model, p, optimizer, loss_functions, train_loader, device = build_training_objects(args)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    profiler_schedule = schedule(
        wait=args.wait,
        warmup=args.warmup,
        active=args.active,
        repeat=args.repeat,
    )
    n_steps = (args.wait + args.warmup + args.active) * args.repeat

    print(f"Profiling {n_steps} training batches on {device}.")
    print(f"Writing TensorBoard traces to: {args.log_dir}")

    with profile(
        activities=activities,
        schedule=profiler_schedule,
        on_trace_ready=tensorboard_trace_handler(str(args.log_dir)),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    ) as prof:
        train_one_profiled_window(
            model=model,
            p=p,
            optimizer=optimizer,
            loss_functions=loss_functions,
            train_loader=train_loader,
            device=device,
            profiler=prof,
            n_steps=n_steps,
        )

    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=25))
    print(f"Open with: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
