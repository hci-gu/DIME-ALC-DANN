import os
import torch
from uuid import uuid4
from utils.early_stopping import EarlyStopping

# TODO create helpre function that returns all name variables that start with "string" for easy config

class Params():

    # Training params
    n_epochs = 30
    lr = 0.001
    batch_size = 32
    n_workers = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_optim_metric = "loss"
    pin_memory = torch.cuda.is_available()
    stopping_criterion = EarlyStopping()

    # Optimizer & Loss funtions & Schedulers


    # Encoder
    encoder_input_dimension = 10
    encoder_hidden_dimension = 10
    encoder_n_layers = 10
    encoder_output_dimension = 10
    encoder_activation_function = "relu"


    # Classifier
    classifier_input_dimension = 10
    classifier_hidden_dimension = 10
    classifier_n_layers = 10
    classifier_output_dimension = 10
    classifier_activation_function = "relu"


    # Discriminator
    discriminator_input_dimension = 10
    discriminator_hidden_dimension = 10
    discriminator_n_layers = 10
    discriminator_output_dimension = 10
    discriminator_activation_function = "relu"