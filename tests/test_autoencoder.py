"""Check the ALC autoencoder evaluation protocol without fitting a model."""

import pytest
import torch

from alc_data import ALCData
from autoencoder import (
    _aggregate_metrics,
    _reconstruction_errors,
    _score_errors,
    _speaker_features,
    _split_sober_indices,
)


def test_sober_split_is_disjoint_and_reproducible():
    sober_indices = list(range(60))
    train, calibration, test = _split_sober_indices(sober_indices, seed=1999, speaker_id=6)

    assert (len(train), len(calibration), len(test)) == (36, 12, 12)
    assert sorted(train + calibration + test) == sober_indices
    assert (train, calibration, test) == _split_sober_indices(
        sober_indices[::-1], seed=1999, speaker_id=6,
    )
    assert train != _split_sober_indices(sober_indices, seed=2000, speaker_id=6)[0]


def test_speaker_standardization_uses_only_sober_training_features():
    data = ALCData.__new__(ALCData)
    data.files = [f"006{index}.wav" for index in range(5)]
    data.tensors_dict = dict(zip(data.files, torch.tensor([
        [1.0, 9.0], [3.0, 9.0], [5.0, 12.0], [6.0, 13.0], [8.0, 14.0],
    ]), strict=True))
    data.class_labels = torch.tensor([0, 0, 0, 0, 1])
    data.bac_values = [0.0, 0.0, 0.0, 0.0, 1.0]
    data.train_speaker_mapping = {}
    data.include_audio_length = False
    data.transforms = None
    data.verbose = False

    train, calibration, test = _speaker_features(data, [0, 1], [2], [3, 4], torch.device("cpu"))

    torch.testing.assert_close(train, torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
    torch.testing.assert_close(calibration, torch.tensor([[3.0, 0.0]]))
    torch.testing.assert_close(test, torch.tensor([[4.0, 0.0], [6.0, 0.0]]))


def test_reconstruction_threshold_and_aggregate_metrics():
    errors = _reconstruction_errors(torch.nn.ReLU(), torch.tensor([[-2.0, 1.0], [-3.0, -1.0]]))
    torch.testing.assert_close(errors, torch.tensor([2.0, 5.0]))

    first = _score_errors(torch.tensor([0.0, 20.0]), torch.tensor([19.0, 20.0, 18.0, 20.0, 18.0]), 3)
    second = _score_errors(torch.tensor([0.0, 10.0]), torch.tensor([0.0, 1.0, 2.0]), 1)

    assert first["threshold"] == 19.0
    assert first == pytest.approx({
        "threshold": 19.0, "tp": 1, "fp": 1, "tn": 2, "fn": 1,
        "precision": 0.5, "recall": 0.5, "f1": 0.5, "balanced_accuracy": 7 / 12,
    })
    assert second["precision"] == second["f1"] == 0.0
    assert _aggregate_metrics([first, second]) == pytest.approx({
        "macro_precision": 0.25, "macro_recall": 0.25,
        "macro_f1": 0.25, "macro_balanced_accuracy": 13 / 24,
        "pooled_precision": 0.5, "pooled_recall": 0.25,
        "pooled_f1": 1 / 3, "pooled_balanced_accuracy": 0.5,
    })
