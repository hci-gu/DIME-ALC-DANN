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
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    optim_metric: str = "loss"
    pin_memory: bool = field(default_factory=lambda: torch.cuda.is_available())


    # Early stopping
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 0.0 
    early_stopping_restore_best_weights: bool = True
    early_stopping_mode: str = "max"

    # Optimizer & Loss funtions & Schedulers


    # Extractor
    extractor_input_dimension: int = 10
    extractor_hidden_dimension: int = 10
    extractor_n_layers: int = 10
    extractor_output_dimension: int = 10
    extractor_activation_function: str = "relu"
    extractor_p_dropout: float = 0.0


    # Classifier
    classifier_input_dimension: int = 10
    classifier_hidden_dimension: int = 10
    classifier_n_layers: int = 10
    classifier_output_dimension: int = 10
    classifier_activation_function: str = "relu"
    classifier_p_dropout: float = 0.0


    # Discriminator
    discriminator_input_dimension: int = 10
    discriminator_hidden_dimension: int = 10
    discriminator_n_layers: int = 10
    discriminator_output_dimension: int = 10
    discriminator_activation_function: str = "relu"
    discriminator_p_dropout: float = 0.0

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