import torch
import torch.nn as nn

from torch import Tensor
from params import Params
from typing import Literal
from torch.nn import Module, functional
from torchsummary import summary
from utils.gradient_reversal import ReverseLayerF
from utils.layer_creation import create_mlp


class DANN(Module):

    def __init__(self, p: Params):
        super().__init__()

        # DANN components
        self.extractor = Extractor(p.get_vars_from_prefix("extractor"))
        self.classifier = Classifier(p.get_vars_from_prefix("classifier"))
        self.discriminator = Discriminator(p.get_vars_from_prefix("discriminator"))
    
    def forward(self, x: Tensor, alpha: float = 1.0) -> tuple[Tensor,Tensor]:
        x = x.squeeze(1) # batch we get from dataloader [B,1,d_input] -> [B,d_input]

        # Extractor
        x_feature = self.extractor(x) # [B,d_e_o]

        # Classifier
        class_logits = self.classifier(x_feature) # [B,d_e_o] -> [B,d_c_o=1]

        # Discriminator
        reversed_feature = ReverseLayerF.apply(x_feature, alpha)
        speaker_logits = self.discriminator(reversed_feature) # [B,d_e_o] -> [B,d_d_o=n_speakers]

        return class_logits, speaker_logits
        
    @torch.no_grad()
    def predict(self, x: Tensor, return_type: Literal["probability","logits"] = "logits") -> Tensor:
        x = x.squeeze(1) # batch we get from dataloader [B,1,D] -> [B,d_input]
        x_feature = self.extractor(x)
        output = self.classifier(x_feature) # logit for positive class \in (-inf,inf)
        if return_type == "probability":
            return functional.sigmoid(output) # p(intoxicated|x) = "probability this person is intoxicated"
        return output


class Extractor(Module):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.extractor = create_mlp(config)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.extractor(x)

class Classifier(Module):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.classifier = create_mlp(config)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)

class Discriminator(Module):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.discriminator = create_mlp(config)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.discriminator(x)

class MLP(Module):

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        p_dropout: float = 0.5,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.BatchNorm1d(hidden_dimension),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


if __name__ == "__main__":
    
    p = Params()
    dann = DANN(p)
    n_features = 6373
    example_input = torch.randn((5,1,n_features))
    summary(dann, example_input, batch_dim=None)
