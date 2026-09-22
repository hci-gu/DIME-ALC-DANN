import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()

        # Derive p_t from the unweighted log probability. Applying class
        # weights before exp(-ce) would turn p_t into p_t ** weight.
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(
            dim=1,
            index=targets.unsqueeze(1),
        ).squeeze(1)
        pt = log_pt.exp()

        if self.weight is None:
            alpha_t = 1.0
        else:
            alpha_t = self.weight[targets]

        return (
            alpha_t
            * (1.0 - pt).pow(self.gamma)
            * (-log_pt)
        ).mean()
