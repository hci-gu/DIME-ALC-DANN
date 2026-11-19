# DIME ALC DANN

Train and evaluate a domain-adversarial neural network (DANN) for speech-based intoxication classification. The project also includes an ALC MLP baseline and live microphone inference.

## Setup

From the repository root, use Python 3.12 and sync the CPU environment with uv.

```bash
uv sync --extra cpu
```

For a GPU compatible with CUDA 13.0, use the CUDA environment.

```bash
uv sync --extra cuda
```

The commands below use the environment selected during setup.

## Datasets and feature caches

| Dataset | Local inputs | Create the OpenSMILE cache |
| --- | --- | --- |
| ALC (`alc`) | WAV files in `data/ALC/wav/h/` and labels in `data/ALC/labels/h/` | `uv run --no-sync python alc_data.py` |
| SE-DAC218 (`dac`) | WAV files in `data/SE-DAC218/audio/wav/` and `data/SE-DAC218/labels/labels.json` | `uv run --no-sync python dac218_data.py` |
| GU (`gu`) | Filtered WAV files in `data/GU/filtered-control/` and `data/GU/filtered-intoxicated/` | `uv run --no-sync python gu_data.py` |

Run the cache command for each dataset you plan to use. The builders extract ComParE 2016 functionals into `.cache/` and require the corresponding cache file to be absent. Keep the source audio and labels available for the dataset loaders.

## Training, finetuning, and evaluation

```bash
uv run --no-sync python main.py --data alc --run-name alc-dann
uv run --no-sync python finetune.py --data gu --checkpoint alc-dann --run-name gu-dann
uv run --no-sync python evaluation.py --data gu --run-name finetuned_gu-dann --eval-split saved-test
uv run --no-sync python evaluation.py --data dac --run-name alc-dann --eval-split all
```

`main.py` and `evaluation.py` default to ALC. `finetune.py` defaults to SE-DAC218 and requires a checkpoint trained on a different dataset. For finetuning and evaluation, specify the filename stem of a checkpoint in `weights/`. Standalone evaluation uses the checkpoint's normalization statistics and decision threshold. The default `--eval-split saved-test` selects the saved test speakers from the checkpoint's dataset. Use `--eval-split all` to evaluate all eligible samples, including when evaluating on a different dataset.

For GU training and finetuning, speakers are split within each cohort so all three splits contain both classes. Each cohort needs at least three speakers. GU has no measured BAC values, so the workflows ignore `--bac-limit` and skip BAC plots. Classification metrics, probability plots, duration diagnostics, and speaker error plots remain available. Per-speaker balanced accuracy is N/A because individual GU speakers have only one class.

Run `uv run --no-sync python baseline.py` for the ALC MLP baseline or `uv run --no-sync python main.py --hpo` for Optuna hyperparameter optimization.

## Microphone inference

Use a DANN checkpoint saved by training or finetuning, including its normalization statistics. Pass its file path to `--checkpoint`.

```bash
uv run --no-sync python inference.py --list-devices
uv run --no-sync python inference.py --checkpoint weights/alc-dann.pth
uv run --no-sync python inference.py --checkpoint weights/alc-dann.pth --window-seconds 10 --input-device 1 --sample-rate 48000
```

The script uses CPU inference and the default microphone at its native sample rate unless
overridden. It continuously records mono audio in consecutive, non-overlapping windows
(5 seconds by default), printing `p(intoxicated)` and RMS in digital sample units
for every complete window, including silence.
Audio stays in memory. Press Ctrl+C to stop and discard unfinished or pending windows.
Capture stops with an error on audio overflow or when a completed window arrives and all
three queue slots are occupied.

## Repository layout

- [`model.py`](model.py), [`train.py`](train.py), and [`params.py`](params.py) define model architectures, training and evaluation routines, and experiment settings.
- [`utils/`](utils/) contains shared CLI, training, and figure-logging utilities.
- [`scripts/`](scripts/) contains data preparation and plots for audio duration and OpenSMILE features.
- The local `data/eda-gu.ipynb` notebook contains GU exploratory analysis and its audio-diagnostics helpers. Git ignores `data/`.
- [`tests/`](tests/) covers GU datasets and workflows, evaluation diagnostics, and microphone inference with fixtures and synthetic inputs.

## Tests

Run the suite from the repository root. The command supplies pytest for that invocation.

```bash
uv run --extra cpu --with pytest python -m pytest
```

Use `--extra cuda` for a CUDA environment.

## MLflow tracking

To serve the tracking UI on the machine running the experiments, start MLflow there.

```bash
uv run --no-sync mlflow server --host 127.0.0.1 --port 5000
```

For a remote machine, forward the port over SSH and open `http://127.0.0.1:5000` in your browser.

```bash
ssh -L 5000:127.0.0.1:5000 user@REMOTE_HOSTNAME
```

### Useful links

- [openSMILE Python documentation](https://audeering.github.io/opensmile-python/)
- [PyTorch DANN example](https://github.com/NaJaeMin92/pytorch-DANN)
