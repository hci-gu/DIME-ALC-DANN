"""Check duration metadata and its alignment with evaluation probabilities."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Subset

from alc_data import ALCData
from dac218_data import DAC218Data
from params import Params
from train import test_evaluation as run_test_evaluation


@pytest.mark.parametrize("dataset_class", [ALCData, DAC218Data])
@pytest.mark.parametrize("include_audio_length", [False, True])
def test_duration_plot_matches_subset_order(
    tmp_path, dataset_class, include_audio_length
):
    # Use real WAV headers without requiring private data or feature caches.
    data = dataset_class.__new__(dataset_class)
    data.AUDIO_PATH = str(tmp_path)
    data.include_audio_length = include_audio_length
    data.files = ["001_short.wav", "002_long.wav", "003_medium.wav"]
    durations = [0.5, 2.0, 1.0]
    for filename, duration in zip(data.files, durations):
        sf.write(tmp_path / filename, np.zeros(int(8000 * duration)), 8000)
    data.speaker_ids = [1, 2, 3]
    data.train_speaker_mapping = {}
    data.class_labels = torch.tensor([0, 1, 0])
    data.bac_values = [0.0, 0.8, 0.0]
    data.tensors_dict = {
        filename: torch.tensor([logit])
        for filename, logit in zip(data.files, [-2.0, 2.0, -1.0])
    }
    data.mu = torch.zeros(1)
    data.sigma = torch.ones(1)
    data.transforms = None
    loader = DataLoader(Subset(data, [1, 0, 2]), batch_size=2)
    plotted_points = {}

    def capture_figure(fig, artifact_path):
        if "audio_length_probability" in artifact_path:
            plotted_points[artifact_path] = np.array(
                fig.axes[0].collections[0].get_offsets()
            )
            assert fig.axes[0].get_xlabel() == "Audio length (s)"

    with patch("utils.figure_logging.mlflow.log_figure", side_effect=capture_figure):
        metrics = run_test_evaluation(
            torch.nn.Identity(), Params(), loader, torch.device("cpu"),
            eval_type="test", threshold=0.5,
        )

    assert metrics["test/accuracy"] == 1.0
    if include_audio_length:
        points = plotted_points["images/test_audio_length_probability_final.png"]
        np.testing.assert_allclose(points[:, 0], [2.0, 0.5, 1.0])
        np.testing.assert_allclose(
            points[:, 1], torch.sigmoid(torch.tensor([2.0, -2.0, -1.0])).numpy()
        )
    else:
        assert not plotted_points
    assert not plt.get_fignums()
