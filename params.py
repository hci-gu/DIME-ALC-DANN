import os
import json
import torch
from dataclasses import dataclass, field, fields, asdict

@dataclass()
class Params():

    dev_run: bool = False # True when developing and doing quick iterations

    # Training params
    n_epochs: int = 50
    batch_size: int = 128
    n_workers: int = 1
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    optim_metric: str = "loss"
    pin_memory: bool = field(default_factory=lambda: torch.cuda.is_available())


    # Early stopping
    early_stopping_patience: int | None = 10
    early_stopping_min_delta: float = 0.0 
    early_stopping_restore_best_weights: bool = True
    early_stopping_mode: str = "min"

    # Optimizer
    optimizer: str = "Adam"
    optimizer_lr: float = 0.001

    # Loss funtions
    use_pos_weight: bool = True
    loss_fn_classifier: str = "BCEWithLogitsLoss"
    loss_fn_discriminator: str = "CrossEntropyLoss"

    # Scheduler
    scheduler: str = "ReduceLROnPlateau"
    scheduler_patience: int = 5
    scheduler_mode: str = "min"

    # Extractor
    extractor_input_dimension: int = 6373
    extractor_hidden_dimension: int = 256
    extractor_n_layers: int = 3
    extractor_output_dimension: int = 256
    extractor_activation_function: str = "relu"
    extractor_p_dropout: float = 0.0


    # Classifier
    classifier_input_dimension: int = extractor_output_dimension
    classifier_hidden_dimension: int = 256
    classifier_n_layers: int = 5
    classifier_output_dimension: int = 1
    classifier_activation_function: str = "relu"
    classifier_p_dropout: float = 0.0


    # Discriminator
    discriminator_input_dimension: int = extractor_output_dimension
    discriminator_hidden_dimension: int = 256
    discriminator_n_layers: int = 5
    discriminator_output_dimension: int = 10
    discriminator_activation_function: str = "relu"
    discriminator_p_dropout: float = 0.0

    @classmethod
    def from_optional_overrides(cls, **overrides):
        field_names = {f.name for f in fields(cls)}
        params_overrides = {
            name: value
            for name, value in overrides.items()
            if name in field_names and value is not None
        }
        return cls(**params_overrides)

    def get_vars_from_prefix(self, prefix: str, strip_prefix: bool = True):
        output = {}

        for f in fields(self):
            name = f.name

            if name.startswith(prefix + "_"):
                value = getattr(self, name)

                if strip_prefix:
                    name = name.removeprefix(prefix + "_")

                output[name] = value

        return output


if __name__ == "__main__":
    p = Params(classifier_hidden_dimension=200)
    classifier_config = p.get_vars_from_prefix("classifier", strip_prefix=True)
    print(classifier_config)

    # with open("testconfig.json", "w") as f:
    #     json.dump(asdict(p), f, indent=4)
