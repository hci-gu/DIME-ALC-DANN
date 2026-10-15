import os
import torch
import torch.nn as nn

from model import DANN
from params import Params

# Load in checkpoint
checkpoint_name = "dann-618dd032.pth"
checkpoint = torch.load(os.path.join("weights", checkpoint_name), map_location="cpu", weights_only=False)
print(checkpoint["params"])
p = Params(**checkpoint["params"])

model = DANN(p)
extractor_params = {k:v for (k,v) in checkpoint["model_state_dict"].items() if k.startswith("extractor")}
classifier_params = {k:v for (k,v) in checkpoint["model_state_dict"].items() if k.startswith("classifier")}
model.load_state_dict(extractor_params | classifier_params)