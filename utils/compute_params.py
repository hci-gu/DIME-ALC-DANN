import math

def alpha_schedule(epoch: int, n_epochs: int, gamma: float = 10.0) -> float:
    p = epoch / max(n_epochs - 1, 1)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0