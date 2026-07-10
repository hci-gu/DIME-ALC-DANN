import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets.long(), weight=self.weight, reduction="none"
        )
        p_t = torch.exp(-ce)
        return ((1 - p_t).pow(self.gamma) * ce).mean()