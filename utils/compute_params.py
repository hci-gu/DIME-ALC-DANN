import math


class AdversialScheduler():

    def __init__(self, n_epochs: int, weight: float, gamma: float = 10, mode: str = "logistic", max_alpha = None):
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.weight = weight # a0
        self.mode = mode
        self.max_alpha = max_alpha

    def __call__(self, epoch):
        if self.mode == "constant":
            return self.weight
        elif self.mode == "logistic":
            return self.alpha_schedule(epoch)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def alpha_schedule(self, epoch: int) -> float:
        p = epoch / max(self.n_epochs - 1, 1)
        if self.max_alpha is None:
            return 2.0 / (1.0 + math.exp(-self.gamma * p)) - 1.0
        else:
            return min(self.max_alpha, 2.0 / (1.0 + math.exp(-self.gamma * p)) - 1.0)