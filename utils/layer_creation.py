import torch
import torch.nn as nn

def get_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }

    name = name.lower()

    if name not in activations:
        raise ValueError(
            f"Unknown activation function: {name}. "
            f"Available options are: {list(activations.keys())}"
        )

    return activations[name]()



def create_mlp(config) -> nn.Sequential:

    if config["n_layers"] < 1:
        raise ValueError("n_layers must be >= 1")

    if not 0.0 <= config["p_dropout"] <= 1.0:
        raise ValueError("dropout_rate must be between 0 and 1")

    layers: list[nn.Module] = []

    if config["n_layers"] == 1:
        layers.append(
            nn.Linear(
                config["input_dimension"],
                config["output_dimension"],
            )
        )
        return nn.Sequential(*layers)

    # Input layer
    layers.append(
        nn.Linear(
            config["input_dimension"],
            config["hidden_dimension"],
        )
    )
    layers.append(get_activation(config["activation_function"]))

    if config["p_dropout"] > 0:
        layers.append(nn.Dropout(config["p_dropout"]))

    # Hidden layers
    for _ in range(config["n_layers"] - 2):
        layers.append(
            nn.Linear(
                config["hidden_dimension"],
                config["hidden_dimension"],
            )
        )
        layers.append(get_activation(config["activation_function"]))

        if config["p_dropout"] > 0:
            layers.append(nn.Dropout(config["p_dropout"]))

    # Output layer
    layers.append(
        nn.Linear(
            config["hidden_dimension"],
            config["output_dimension"],
        )
    )

    return nn.Sequential(*layers)