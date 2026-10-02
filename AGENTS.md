# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python 3.12 research project for training a domain-adversarial neural network on ALC and SE-DAC218 speech data. The main workflow starts in `main.py`. Model architecture lives in `model.py`, training and evaluation routines in `train.py`, dataset handling in `alc_data.py` and `dac218_data.py`, and experiment settings in `params.py`. `baseline.py`, `finetune.py`, and `evaluation.py` provide adjacent workflows. Reusable components belong in `utils/`; data preparation, profiling, and benchmarks belong in `scripts/`.

Large or generated artifacts (`data/`, `.cache/`, `weights/`, `evals/`, `mlruns/`, TensorBoard output, and `mlflow.db`) are intentionally ignored and must not be committed.

## Build, Test, and Development Commands

- `uv sync --extra cpu` installs the locked CPU environment.
- `uv sync --extra cuda` installs PyTorch from the CUDA 13.0 index for compatible systems.
- `uv run python alc_data.py` or `uv run python dac218_data.py` creates the corresponding OpenSMILE feature cache from local audio. Each command expects its cache file not to exist yet.
- `uv run python main.py --dev-run --n-epochs 1 --no-save-model --disable-logs` runs a training smoke test using the ALC data. `--dev-run` limits training batches but still loads the full dataset and evaluates the full validation and test splits.
- `uv run python main.py --hpo` launches Optuna hyperparameter optimization; expect a long-running job.
- `uv run mlflow server --host 127.0.0.1 --port 5000` serves local experiment tracking.
- `uv run tensorboard --logdir scripts/profiling/tensorboard` displays profiler traces.

Use `uv run python <script>` so commands execute in the locked environment. `main.py --data dac` selects SE-DAC218; ALC is the default. `--disable-logs` routes MLflow output to a temporary store for that invocation. There is no separate build step.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for modules, functions, variables, and CLI flags; use `PascalCase` for classes such as `DANN` and `ALCData`; reserve uppercase names for true constants. Add type hints to new public functions and keep imports grouped as standard library, third-party, then local modules. No formatter or linter is currently configured, so match nearby code and keep changes focused.

## Testing Guidelines

No automated test suite or coverage threshold exists yet. For every change, run a focused smoke test and compile affected modules with `uv run python -m py_compile <files>`. New tests should use `pytest`, live under `tests/`, and follow `test_<module>.py` and `test_<behavior>()` naming. Avoid tests that require the full private dataset; use small fixtures or synthetic tensors.

## Commit & Pull Request Guidelines

History uses short, lowercase summaries such as `bug fixes` and `param log bug v2`. Keep commits concise and single-purpose, but prefer an imperative description such as `fix HPO parameter logging`. Pull requests should explain the motivation and behavior change, list verification commands, link relevant issues, and note data, GPU, or MLflow requirements. Include plots or metric comparisons when training behavior changes.
