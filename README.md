# DIME ALC DANN

## Setup 

For CPU-only devices with no GPU/CUDA 
``` bash
uv sync --extra cpu
```

For CUDA/GPU compatible devices (recommended)
``` bash
uv sync --extra cuda
```

## Mlflow (remote serving)
Serve the model on remote machine
``` bash
 uv run mlflow server --host 127.0.0.1 --port 5000
```

Create a temporary port forward on your local machine
``` bash
ssh -L 5000:127.0.0.1:5000 user@REMOTE_HOSTNAME
```

## Tensorboard
Start tensorboard server on remote machine
```
uv run tensorboard --logdir profiling\tensorboard
```

Create a temporary port forward on your local machine
``` bash
ssh -L 5000:127.0.0.1:5000 user@REMOTE_HOSTNAME
```