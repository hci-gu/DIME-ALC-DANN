import argparse

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DANN trainer")
    parser.add_argument("--dev-run", action="store_true", default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--save-model", action="store_true", default=True)
    parser.add_argument("--use-pos-weight", action="store_true", default=True)
    parser.add_argument("--cache-features", action="store_true", default=True)
    parser.add_argument("--batch-size", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=1999)
    parser.add_argument("--n-workers", "--num-workers", dest="n_workers", type=non_negative_int, default=None)
    parser.add_argument("--n-epochs", "--num-epochs", dest="n_epochs", type=positive_int, default=None)
    parser.add_argument("--max-samples", type=positive_int, default=None)

    return parser.parse_args()
