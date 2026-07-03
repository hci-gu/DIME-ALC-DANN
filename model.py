import torch
import torch.nn as nn

from params import Params
from torch.nn import Module, functional
from torchsummary import summary
from utils.gradient_reversal import ReverseLayerF
from utils.layer_creation import create_mlp


class DANN(Module):

    def __init__(self, p: Params):
        super().__init__()

        self.extractor = Extractor(p.get_vars_from_prefix("extractor"))
        self.classifier = Classifier(p.get_vars_from_prefix("classifier"))
        self.discriminator = Discriminator(p.get_vars_from_prefix("discriminator"))
    
    def forward(self, x, alpha: float = 1.0):
        x = x # batch we get from dataloader [B,1,D]

        # Extractor
        x_feature = self.extractor(x)

        # Classifier
        class_logits = self.classifier(x_feature)

        # Discriminator
        reversed_feature = ReverseLayerF.apply(x_feature, alpha)
        speaker_logits = self.discriminator(reversed_feature)

        return class_logits, speaker_logits
        
    @torch.no_grad()
    def predict(self, x):
        x = x # batch we get from dataloader [B,1,D]
        x_feature = self.extractor(x)
        logit = self.classifier(x_feature)
        p_intoxicated = functional.sigmoid(logit) # p(intoxicated|x) = "probability this person is intoxicated"
        return p_intoxicated


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
    n_features = 6373
    example_input = torch.randn((5,1,n_features))
    summary(dann, example_input, batch_dim=None)