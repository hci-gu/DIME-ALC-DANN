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

## Microphone inference

Use a DANN checkpoint saved by training or finetuning (including its normalization statistics):

``` bash
uv run python inference.py --list-devices
uv run python inference.py --checkpoint weights/my-model.pth
uv run python inference.py --checkpoint weights/my-model.pth --window-seconds 5 --input-device 1 --sample-rate 48000
```

The script uses CPU inference and the default microphone at its native sample rate unless
overridden. It continuously records mono audio in consecutive, non-overlapping windows
(10 seconds by default), printing `p(intoxicated)` and RMS for every complete window,
including silence. RMS is measured in digital sample units, not calibrated sound pressure.
Audio stays in memory. Press Ctrl+C to stop and discard unfinished or pending windows.
Capture stops with an error if audio overflows or processing falls more than three queued
windows behind.

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

### Useful links:
- https://audeering.github.io/opensmile-python/ (OpenSmile)
- https://github.com/NaJaeMin92/pytorch-DANN (pytorch DANN example)
