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

## ALC per-speaker autoencoders

On the DGX Spark, use the CUDA environment from [Setup](#setup). If `.cache/alc-opensmile-features.pt` is absent, build it first with `uv run --no-sync python alc_data.py`. Keep the ALC headset WAV files and labels in their usual locations.

```bash
uv run --no-sync python autoencoder.py --device cuda --run-name alc-speaker-ae
```

The script uses a seeded 60/20/20 split of each speaker's sober utterances. It fits feature standardization and the MLP on the training split, sets the reconstruction-error threshold at the 95th percentile of the calibration split, and tests on the remaining sober utterances and all intoxicated utterances.

The MLflow experiment `autoencoder` records one run per invocation. Download `speaker_metrics.csv` for per-speaker thresholds, counts, precision, recall, F1, and balanced accuracy. Run metrics prefixed with `macro_` weight speakers equally. Metrics prefixed with `pooled_` combine their test confusion counts before calculating each score. For reconstruction models, `--device auto` selects CUDA when available and uses CPU otherwise.

### Compare recall at a common false-positive target

Run each comparison with the same seed and `--target-fpr` so the models use the same speaker cohorts, utterance splits, and development false-positive limit. The commands below use the default seed of 1999 and a target of 15%. Build the separate 88-feature eGeMAPS cache once for the eGeMAPS comparison.

```bash
uv run --no-sync python alc_data.py --feature-set eGeMAPSv02
uv run --no-sync python autoencoder.py --threshold-sweep --target-fpr 0.15 --device cuda --run-name compare-mlp
uv run --no-sync python autoencoder.py --threshold-sweep --target-fpr 0.15 --method pca --device cuda --run-name compare-pca
uv run --no-sync python autoencoder.py --threshold-sweep --target-fpr 0.15 --feature-set eGeMAPSv02 --device cuda --run-name egemaps-mlp
uv run --no-sync python autoencoder.py --threshold-sweep --target-fpr 0.15 --method shared-mlp --device cuda --run-name compare-shared-mlp
```

The sweep divides eligible speakers into a shared-model training cohort, a development cohort, and a final evaluation cohort in a seeded 60/20/20 split. PCA defaults to eight components and fits on each speaker's sober training utterances. The optional shared MLP fits on the training cohort's standardized sober training utterances. It uses each evaluation speaker's own sober training subset for standardization and sober calibration subset for the cutoff.

`threshold_sweep.csv` reports the development results for sober calibration percentiles 80, 85, 90, and 95. The selected percentile gives the highest pooled development recall within the target sober false-positive rate. If none qualifies, the rule predicts sober for every utterance. Compare `pooled_recall` and the achieved `pooled_sober_fpr` together on the final cohort. The development target and final false-positive rate can differ. For a comparison at exactly the same target, `macro_roc_recall_at_target_fpr` interpolates each final speaker's ROC curve and averages their recall. This diagnostic uses test labels to evaluate score ranking and is independent of cutoff selection.

`utterance_scores.csv` contains the calibration and test reconstruction errors, with their speaker cohort and class labels. `matched_prompts.csv` lists paired sober and intoxicated test utterances linked by the ALC annotations. MLflow logs how often the intoxicated utterance scores higher in the final cohort. `speaker_partitions.json` records the cohort assignments. The original command without `--threshold-sweep` evaluates all eligible speakers at the fixed 95th percentile and also logs the score and prompt diagnostics.

### Per-speaker logistic regression

From the repository root, build `.cache/alc-egemaps-features.pt` once if it is absent.

```bash
uv run --no-sync python alc_data.py --feature-set eGeMAPSv02
```

Fit one L2-regularized logistic regression per speaker. The model uses OpenSMILE features and applies the same speaker-specific standardization to both classes, fitted on the speaker's sober training utterances. Each fit uses 36 sober and 18 intoxicated utterances on the full ALC dataset. The cutoff is calibrated on 12 separate sober utterances, and the test set has 12 sober and 12 intoxicated utterances per speaker. The protocol uses labeled intoxicated enrollment from each evaluated speaker and holds out utterances within the ALC recording sessions.

```bash
uv run --no-sync python autoencoder.py --method logistic --feature-set eGeMAPSv02 --threshold-sweep --target-fpr 0.15 --device cpu --run-name egemaps-speaker-lr
uv run --no-sync python autoencoder.py --method mlp --feature-set eGeMAPSv02 --threshold-sweep --target-fpr 0.15 --reserve-intoxicated-training --device cuda --run-name egemaps-mlp-same-test
```

The second command evaluates the sober-only MLP on the same held-out utterances when the seed matches. `--lr-c` controls inverse regularization strength and defaults to `0.1`. Logistic regression uses class-balanced training weights and logs `decision_score` in `utterance_scores.csv`. Compare `macro_roc_recall_at_target_fpr`, `pooled_recall`, and `pooled_sober_fpr` in MLflow.

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
