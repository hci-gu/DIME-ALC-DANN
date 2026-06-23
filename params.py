import os
import json
import torch

from uuid import uuid4
from dataclasses import dataclass, field, fields, asdict
from utils.early_stopping import EarlyStopping

@dataclass(frozen=True)
class Params():

    # Training params
    n_epochs: int = 30
    lr: float = 0.001
    batch_size: int = 32
    n_workers: int = 1
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    p_optim_metric: str = "loss"
    pin_memory: bool = field(default_factory=lambda: torch.cuda.is_available())
    stopping_criterion: EarlyStopping = field(default_factory=EarlyStopping)

    # Optimizer & Loss funtions & Schedulers


    # Encoder
    encoder_input_dimension: int = 10
    encoder_hidden_dimension: int = 10
    encoder_n_layers: int = 10
    encoder_output_dimension: int = 10
    encoder_activation_function: str = "relu"


    # Classifier
    classifier_input_dimension: int = 10
    classifier_hidden_dimension: int = 10
    classifier_n_layers: int = 10
    classifier_output_dimension: int = 10
    classifier_activation_function: str = "relu"


    # Discriminator
    discriminator_input_dimension: int = 10
    discriminator_hidden_dimension: int = 10
    discriminator_n_layers: int = 10
    discriminator_output_dimension: int = 10
    discriminator_activation_function: str = "relu"

    def get_vars_from_prefix(self, prefix: str, strip_prefix: bool = False):
        output = {}

        for f in fields(self):
            name = f.name

            if name.startswith(prefix):
                value = getattr(self, name)

                if strip_prefix:
                    name = name.removeprefix(prefix + "_")

                output[name] = value

        return output


if __name__ == "__main__":
    p = Params(classifier_hidden_dimension=200)
    classifier_config = p.get_vars_from_prefix("classifier", strip_prefix=True)
    print(classifier_config)

    with open("testconfig.json", "w") as f:
        json.dump(asdict(p), f, indent=4)