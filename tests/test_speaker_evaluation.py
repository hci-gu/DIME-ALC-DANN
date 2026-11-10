"""Verify per-speaker balanced accuracy through the test evaluation path."""

from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from params import Params
from train import test_evaluation as run_test_evaluation


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
