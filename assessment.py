import torch
from model import DANN
from torch.utils.data import DataLoader

@torch.no_grad()
def assess_model(model: DANN, p:Params, classifier_loss_fn, eval_loader: DataLoader, device, epoch: int = None, eval_type: Literal["val","test"] = "val") -> dict:
    pass