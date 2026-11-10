"""Verify per-speaker balanced accuracy through the test evaluation path."""

from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from params import Params
from train import test_evaluation as run_test_evaluation
from utils.figure_logging import _log_speaker_confident_errors


def test_speaker_balanced_accuracy_uses_shared_threshold():
    # Interleave speakers across batches; speaker 10 has an imbalanced split.
    samples = [
        (20, 1, 0.7), (10, 0, 0.1), (30, 1, 0.9),
        (10, 0, 0.2), (20, 0, 0.6), (10, 1, 0.7),
        (10, 0, 0.3),
    ]
    dataset = [
        (
            torch.logit(torch.tensor([probability])), label,
            {"speaker_id": speaker, "bac": 0.8 if label else 0.0},
        )
        for speaker, label, probability in samples
    ]
    captured = {}

    def capture_figure(fig, artifact_path):
        if "speaker_balanced_accuracy" in artifact_path:
            captured["path"] = artifact_path
            ax = fig.axes[0]
            captured["labels"] = [tick.get_text() for tick in ax.get_xticklabels()]
            captured["scores"] = [bar.get_height() for bar in ax.patches]
            captured["annotations"] = [text.get_text() for text in ax.texts]

    with patch("utils.figure_logging.mlflow.log_figure", side_effect=capture_figure):
        run_test_evaluation(
            torch.nn.Identity(), Params(), DataLoader(dataset, batch_size=2),
            torch.device("cpu"), eval_type="test", threshold=0.8,
        )

    assert captured["path"] == "images/test_speaker_balanced_accuracy_final.png"
    assert captured["labels"] == ["10", "20", "30"]
    np.testing.assert_allclose(captured["scores"], [0.5, 0.5, np.nan])
    assert captured["annotations"] == ["N/A"]


def test_confident_error_counts_exclude_boundaries_and_keep_all_speakers():
    # Include exact boundaries, their adjacent floats, and confident correct cases.
    speakers = np.array([20, 10, 20, 10, 30, 30, 20, 10, 20, 10])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    probabilities = np.array([
        0.95, 0.05, 0.9, 0.1, 0.0, 1.0,
        np.nextafter(0.9, 1.0), np.nextafter(0.1, 0.0), 1.0, 0.0,
    ])
    with patch("utils.figure_logging.mlflow.log_figure") as log_figure:
        _log_speaker_confident_errors(labels, probabilities, speakers, "test", None)

    fig, artifact_path = log_figure.call_args.args
    assert artifact_path == "images/test_speaker_confident_errors_final.png"
    ax = fig.axes[0]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["10", "20", "30"]
    np.testing.assert_array_equal(
        [bar.get_height() for bar in ax.containers[0]], [0, 3, 0]
    )
    np.testing.assert_array_equal(
        [bar.get_height() for bar in ax.containers[1]], [3, 0, 0]
    )
    assert [text.get_text() for text in ax.texts] == ["3", "3", "0"]


def test_confident_errors_logged_by_test_evaluation():
    samples = [(20, 0, 0.95), (10, 1, 0.05), (30, 0, 0.5), (20, 0, 0.99)]
    dataset = [
        (
            torch.logit(torch.tensor([probability])), label,
            {"speaker_id": speaker, "bac": 0.8 if label else 0.0},
        )
        for speaker, label, probability in samples
    ]
    with patch("utils.figure_logging.mlflow.log_figure") as log_figure:
        run_test_evaluation(
            torch.nn.Identity(), Params(), DataLoader(dataset, batch_size=2),
            torch.device("cpu"), eval_type="test", threshold=0.229,
        )

    figures = {call.args[1]: call.args[0] for call in log_figure.call_args_list}
    ax = figures["images/test_speaker_confident_errors_final.png"].axes[0]
    assert [text.get_text() for text in ax.texts] == ["1", "2", "0"]
    ax = figures["images/test_speaker_confident_errors_percentage_final.png"].axes[0]
    assert [text.get_text() for text in ax.texts] == ["100.0%", "100.0%", "0.0%"]


def test_confident_error_percentages_use_total_samples_per_speaker():
    speakers = np.array([20, 10, 30, 10, 20, 10, 10])
    labels = np.array([0, 0, 1, 1, 1, 0, 1])
    probabilities = np.array([0.95, 0.95, 0.99, 0.05, 0.99, 0.01, 0.99])
    with patch("utils.figure_logging.mlflow.log_figure") as log_figure:
        _log_speaker_confident_errors(
            labels, probabilities, speakers, "test", None, normalize=True
        )

    fig, artifact_path = log_figure.call_args.args
    assert artifact_path == "images/test_speaker_confident_errors_percentage_final.png"
    ax = fig.axes[0]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "10\nn=4", "20\nn=2", "30\nn=1"
    ]
    np.testing.assert_allclose(
        [bar.get_height() for bar in ax.containers[0]], [25, 50, 0]
    )
    np.testing.assert_allclose(
        [bar.get_height() for bar in ax.containers[1]], [25, 0, 0]
    )
    np.testing.assert_allclose(
        [bar.get_y() for bar in ax.containers[1]], [25, 50, 0]
    )
    assert [text.get_text() for text in ax.texts] == ["50.0%", "50.0%", "0.0%"]
