# Repository Guidelines

## Datasets

- ALC (`--data alc`) is the default for DANN training and standalone evaluation and is used by the MLP baseline.
- SE-DAC218 (`--data dac`) is the default finetuning target and also supports training and evaluation.
- GU (`--data gu`) is an internal dataset supported by training, finetuning, and evaluation. Its two cohorts provide class labels. GU has no measured BAC values.

## Project Structure & Module Organization

This Python 3.12 research project trains domain-adversarial neural networks (DANNs) for speech-based intoxication classification.

- `main.py` trains a DANN or runs Optuna hyperparameter optimization. `baseline.py` trains the ALC MLP baseline.
- `finetune.py` adapts a DANN checkpoint to a different dataset. `evaluation.py` evaluates saved DANN checkpoints, and `inference.py` runs microphone inference on CPU.
- `model.py` defines the models, `train.py` contains training and evaluation routines, and `params.py` defines experiment settings.
- `alc_data.py`, `dac218_data.py`, and `gu_data.py` contain dataset loaders and OpenSMILE cache builders.
- `utils/` contains shared components such as CLI parsing and evaluation figures. `scripts/` contains data preparation and exploratory analysis tools. `tests/` contains the pytest suite.

Large or generated artifacts (`data/`, `.cache/`, `weights/`, `evals/`, `mlruns/`, TensorBoard output, and `mlflow.db`) are intentionally ignored and must not be committed.

## Build, Test, and Development Commands

- `uv sync --extra cpu` installs the locked CPU environment.
- `uv sync --extra cuda` installs PyTorch from the CUDA 13.0 index for compatible systems.
- `uv run --no-sync python alc_data.py`, `uv run --no-sync python dac218_data.py`, and `uv run --no-sync python gu_data.py` create their respective OpenSMILE feature caches from local audio. Each command requires an unused cache path.
- `uv run --no-sync python main.py --dev-run --n-epochs 1 --no-save-model --disable-logs` runs a training smoke test using ALC. It limits training batches and still loads the full dataset and evaluates the full validation and test splits.
- `uv run --no-sync python main.py --hpo` launches Optuna hyperparameter optimization and can take a long time.
- `uv run --no-sync mlflow server --host 127.0.0.1 --port 5000` serves experiment tracking.

Run commands from the repository root after syncing the CPU or CUDA environment. `--disable-logs` routes training or finetuning MLflow output to a temporary store for that invocation. See the README for dataset paths and workflow examples.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for modules, functions, and variables, and use hyphenated CLI flags such as `--batch-size`. Use `PascalCase` for classes and reserve uppercase names for constants. Add type hints to new public functions and group imports as standard library, third-party, then local modules. Match nearby code and keep changes focused.

## Testing Guidelines

The pytest suite covers GU data and workflows, evaluation diagnostics, and microphone inference. Run it with `uv run --extra cpu --with pytest python -m pytest`, or substitute `--extra cuda` for a CUDA environment. Append a path under `tests/` to run a focused subset. For Python changes, run the relevant tests or smoke test and compile affected modules with `uv run --no-sync python -m py_compile <files>`. Name new tests `tests/test_<module>.py` and `test_<behavior>()`. Use small fixtures or synthetic tensors so tests can run without private datasets.

## Commit & Pull Request Guidelines

History uses short, lowercase summaries such as `bug fixes` and `param log bug v2`. Keep commits concise and single-purpose, but prefer an imperative description such as `fix HPO parameter logging`. Pull requests should explain the motivation and behavior change, list verification commands, link relevant issues, and note data, GPU, or MLflow requirements. Include plots or metric comparisons when training behavior changes.
