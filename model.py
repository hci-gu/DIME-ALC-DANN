import torch
import torch.nn as nn

from params import Params
from torch.nn import Module
from utils.gradient_reversal import ReverseLayerF


class DANN(Module):

    def __init__(self, p: Params):
        super().__init__()

        self.encoder = Encoder()

        self.classifier = Classifier()

        self.gradient_reversal = ReverseLayerF()

        self.discriminator = Discriminator()
    
    def forward(self, x):
        pass

    def predict(self, x):
        pass


class Encoder(Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass

class Classifier(Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass

class Discriminator(Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass


if __name__ == "__main__":
    
    p = Params()
    dann = DANN(p)