"""Exercise GU training, finetuning, and evaluation without private data."""

from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import soundfile as sf
import torch

import evaluation
import finetune
import main as training
from gu_data import GUData, GU_FOLDERS, hash_audio_file
from params import Params
from utils.argument_parsing import parse_args


@pytest.fixture
def gu_dataset(tmp_path):
    root = tmp_path / "GU"
    tensors, hashes = {}, {}
    for label, folder in enumerate(GU_FOLDERS):
        directory = root / folder
        directory.mkdir(parents=True)
        for speaker in range(4):
            for utterance in range(2):
                path = directory / f"21_FEMALE_{speaker}_{utterance}.wav"
                sf.write(path, np.zeros(800), 8000)
                key = path.relative_to(root).as_posix()
                tensors[key] = torch.tensor([label, speaker, utterance, 1.0])
                hashes[key] = hash_audio_file(path)
    cache_path = tmp_path / "features.pt"
    torch.save({
        "tensors": tensors, "hashes": hashes,
        "feature_set": "ComParE_2016", "feature_level": "Functionals",
    }, cache_path)
    return partial(GUData, data_path=root, cache_path=cache_path)


def test_gu_splits_preserve_classes_and_speakers(gu_dataset):
    data = gu_dataset(seed=7)
    splits = data.speaker_split(0.7, 0.15, 0.15)
    speakers = [set(data.speaker_ids[index] for index in indices) for indices in splits]
    assert not speakers[0] & speakers[1]
    assert not speakers[0] & speakers[2]
    assert not speakers[1] & speakers[2]
    assert sorted(index for indices in splits for index in indices) == list(range(len(data)))
    for indices in splits:
        assert set(data.class_labels[indices].tolist()) == {0, 1}
    assert gu_dataset(seed=7).speaker_split(0.7, 0.15, 0.15) == splits
    data.calculate_mu_sigma(splits[0])
    for index in splits[0]:
        assert 0 <= data[index][2]["local_index"] < len(data.train_speakers_id)
    for index in splits[1] + splits[2]:
        assert data[index][2]["local_index"] == -1


def test_gu_training_finetuning_and_checkpoint_evaluation(
    gu_dataset, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for module in (training, finetune, evaluation):
        monkeypatch.setattr(module, "GUData", gu_dataset)
    params = Params(
        device="cpu", pin_memory=False, n_workers=0, n_epochs=1,
        extractor_input_dimension=4, extractor_output_dimension=2,
        extractor_n_layers=1, classifier_n_layers=1, discriminator_n_layers=1,
    )
    monkeypatch.setattr(training, "Params", lambda: replace(params))
    for name in ("set_experiment", "log_metric", "log_metrics", "log_dict", "log_params"):
        monkeypatch.setattr(training.mlflow, name, MagicMock())
    monkeypatch.setattr(training.mlflow, "get_tracking_uri", lambda: "test")
    monkeypatch.setattr(training.mlflow, "start_run", MagicMock())
    active = SimpleNamespace(data=SimpleNamespace(tags={"mlflow.runName": "gu-trained"}))
    monkeypatch.setattr(training.mlflow, "active_run", lambda: active)
    artifacts = []
    monkeypatch.setattr(
        training.mlflow, "log_figure", lambda fig, path: artifacts.append(path)
    )

    def args_for(profile, *extra):
        monkeypatch.setattr("sys.argv", [
            "test", "--data", "gu", "--seed", "7", "--bac-limit", "0.5", *extra
        ])
        return parse_args(profile)

    training.main(args_for("main", "--n-epochs", "1", "--n-workers", "0"))
    trained = torch.load("weights/gu-trained.pth", weights_only=False)
    assert trained["dataset"] == "gu"
    assert trained["bac_limit"] is None
    assert set(trained["speaker_splits"]) == {"train_speakers", "val_speakers", "test_speakers"}

    # Provide a separate source dataset identity for transfer learning.
    source = {**trained, "dataset": "alc", "bac_limit": 0.3}
    torch.save(source, "weights/source.pth")
    active.data.tags["mlflow.runName"] = "gu-finetuned"
    finetune.main(args_for("finetune", "--checkpoint", "source"))
    checkpoint = torch.load("weights/finetuned_gu-finetuned.pth", weights_only=False)
    assert checkpoint["dataset"] == "gu"
    assert checkpoint["bac_limit"] is None
    assert checkpoint["params"]["discriminator_output_dimension"] == 4
    logged = [call.args[0] for call in training.mlflow.log_params.call_args_list]
    assert all(entry["dataset"] == "gu" and entry["lower_bac_limit"] is None for entry in logged)

    actual_evaluation = evaluation.test_evaluation
    evaluated_sizes = []

    def inspect_evaluation(**kwargs):
        subset = kwargs["eval_loader"].dataset
        evaluated_sizes.append(len(subset))
        assert kwargs["bac_limit"] is None
        assert kwargs["threshold"] == checkpoint["threshold"]
        torch.testing.assert_close(subset.dataset.mu, checkpoint["mu"])
        torch.testing.assert_close(subset.dataset.sigma, checkpoint["sigma"])
        metrics = actual_evaluation(**kwargs)
        assert all(np.isfinite(value) for value in metrics.values())
        return metrics

    monkeypatch.setattr(evaluation, "test_evaluation", inspect_evaluation)
    for split in ("saved-test", "all"):
        args_for("evaluation", "--run-name", "finetuned_gu-finetuned", "--eval-split", split)
        evaluation.main()
    assert evaluated_sizes == [4, 16]
    assert any("confusion_matrix" in path for path in artifacts)
    assert any("audio_length_performance" in path for path in artifacts)
    assert any("speaker_confident_errors" in path for path in artifacts)
    assert not any("bac" in path for path in artifacts)
    assert not plt.get_fignums()
