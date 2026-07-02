import torch
import torch.nn as nn

from params import Params
from torch.nn import Module
from torchsummary import summary
from utils.gradient_reversal import ReverseLayerF
from utils.layer_creation import create_mlp


class DANN(Module):

    def __init__(self, p: Params):
        super().__init__()

        self.extractor = Extractor(p.get_vars_from_prefix("extractor", strip_prefix=True))

        self.classifier = Classifier(p.get_vars_from_prefix("classifier", strip_prefix=True))

        self.gradient_reversal = ReverseLayerF()

        self.discriminator = Discriminator(p.get_vars_from_prefix("discriminator", strip_prefix=True))
    
    def forward(self, x):
        x = x # batch we get from dataloader [B,1,D]
        x_feature = self.extractor(x)
        x_classifier = self.classifier(x_feature)
        x_discriminator = self.discriminator(x_feature)

        return x_classifier, x_discriminator
        

    def predict(self, x):
        x = x # batch we get from dataloader [B,1,D]
        x_feature = self.extractor(x)
        x_classifier = self.classifier(x_feature)
        return x_classifier


class Extractor(Module):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.extractor = create_mlp(config)
    
    def forward(self, x):
        x = self.extractor(x)
        return x

class Classifier(Module):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.classifier = create_mlp(config)
    
    def forward(self, x):
        x = self.classifier(x)
        return x

class Discriminator(Module):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.discriminator = create_mlp(config)
    
    def forward(self, x):
        x = self.discriminator(x)
        return x


if __name__ == "__main__":
    
    p = Params()
    dann = DANN(p)
    example_input = torch.randn((5,1,10))
    summary(dann, example_input, batch_dim=None)