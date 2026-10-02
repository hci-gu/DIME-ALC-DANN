import argparse
from typing import Literal

ArgProfile = Literal["main", "finetune", "evaluation"]

def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative float")
    return parsed

def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed

def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed

def parse_args(profile: ArgProfile = "main") -> argparse.Namespace:

    if profile == "main":
        parser = argparse.ArgumentParser(description="DANN trainer")
        parser.add_argument("--dev-run", action="store_true", default=None)
        parser.add_argument("--hpo", action="store_true", default=False)
        parser.add_argument("--disable-logs", action="store_true", default=False)
        parser.add_argument("--verbose", action="store_true", default=False)
        parser.add_argument("--run-name", type=str, default=None)
        parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--use-pos-weight", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--batch-size", type=positive_int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--n-workers", "--num-workers", dest="n_workers", type=non_negative_int, default=None)
        parser.add_argument("--n-epochs", "--num-epochs", dest="n_epochs", type=positive_int, default=None)
        parser.add_argument("--bac-limit", type=non_negative_float, default=None)
        parser.add_argument("--data", choices=["alc", "dac"], default="alc")
    elif profile == "finetune":
        parser = argparse.ArgumentParser(description="DANN trainer")
        parser.add_argument("--dev-run", action="store_true", default=None)
        parser.add_argument("--disable-logs", action="store_true", default=False)
        parser.add_argument("--verbose", action="store_true", default=False)
        parser.add_argument("--run-name", type=str, default=None)
        parser.add_argument("--checkpoint", type=str, default=None)
        parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--use-pos-weight", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--batch-size", type=positive_int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--n-workers", "--num-workers", dest="n_workers", type=non_negative_int, default=None)
        parser.add_argument("--n-epochs", "--num-epochs", dest="n_epochs", type=positive_int, default=None)
        parser.add_argument("--bac-limit", type=non_negative_float, default=None)
    elif profile == "evaluation":
        parser = argparse.ArgumentParser(description="Run evaluation for a logged DANN model")
        parser.add_argument("--run-name", type=str, required=False)
        parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--batch-size", type=positive_int, default=None)
        parser.add_argument("--bac-limit", type=non_negative_float, default=None)
        parser.add_argument("--verbose", action="store_true", default=False)
        parser.add_argument("--data", choices=["alc", "dac"], default="alc")
    else:
        raise ValueError(f"Unknown argument parser profile: {profile}")

    return parser.parse_args()
