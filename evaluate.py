import torch
from model import DANN

@torch.no_grad()
def eval(model: DANN, eval_loader):
    model.eval()
    pass

@torch.no_grad()
def test_evaluation(model: DANN, eval_loader):
    model.eval()
    pass