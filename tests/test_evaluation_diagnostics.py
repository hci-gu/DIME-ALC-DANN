"""Check test diagnostics against known class counts and prediction outcomes."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from params import Params
from train import test_evaluation as run_test_evaluation


def test_diagnostic_figures_use_evaluation_threshold_and_bin_boundaries(tmp_path):
    labels = [0, 1, 1, 0, 1, 0]
    probabilities = [0.1, 0.7, 0.9, 0.85, 0.95, 0.2]
    durations = [1.0, 1.0, 2.0, 2.0, 20.0, 20.0]
    bac_values = [0.0, 0.2, 0.4, 0.0, 0.6, 0.0]
    dataset = [
        (
            torch.logit(torch.tensor([probability])), label,
            {"speaker_id": index % 2, "bac": bac, "audio_length": duration},
        )
        for index, (label, probability, duration, bac) in enumerate(
            zip(labels, probabilities, durations, bac_values)
        )
    ]
    captured = {}

    def capture_figure(fig, artifact_path):
        key = artifact_path.removeprefix("images/test_").removesuffix("_final.png")
        if key in {"probability_histogram", "audio_length_performance", "bac_sensitivity"}:
            captured[key] = fig
            fig.savefig(tmp_path / f"{key}.png")

    with patch("utils.figure_logging.mlflow.log_figure", side_effect=capture_figure):
        run_test_evaluation(
            torch.nn.Identity(), Params(), DataLoader(dataset, batch_size=2),
            torch.device("cpu"), eval_type="test", threshold=0.8, bac_limit=0.35,
        )

    hist_ax = captured["probability_histogram"].axes[0]
    # Both classes use identical bins and retain all observations.
    assert [sum(bar.get_height() for bar in group) for group in hist_ax.containers] == [3, 3]
    np.testing.assert_allclose(hist_ax.lines[0].get_xdata(), [0.8, 0.8])
    assert [text.get_text() for text in hist_ax.get_legend().get_texts()][:2] == [
        "Sober (n=3)", "Intoxicated (n=3)"
    ]

    duration_ax = captured["audio_length_performance"].axes[0]
    for bars, expected in zip(duration_ax.containers, [
        [0.5, 0.5, np.nan, np.nan, 1.0],
        [0.0, 1.0, np.nan, np.nan, 1.0],
        [1.0, 0.0, np.nan, np.nan, 1.0],
    ]):
        np.testing.assert_allclose([bar.get_height() for bar in bars], expected)
    assert "S=1, I=1" in duration_ax.get_xticklabels()[1].get_text()
    assert "S=0, I=0" in duration_ax.get_xticklabels()[2].get_text()

    bac_ax, sober_ax = captured["bac_sensitivity"].axes
    np.testing.assert_allclose(
        [bar.get_height() for bar in bac_ax.patches], [np.nan, 0, np.nan, 1]
    )
    np.testing.assert_allclose(bac_ax.lines[0].get_xdata(), [0.35, 0.35])
    assert [text.get_text() for text in bac_ax.texts] == ["N/A\nn=0", "n=1", "N/A\nn=0", "n=2"]
    assert sober_ax.patches[0].get_height() == pytest.approx(2 / 3)
    assert not plt.get_fignums()


@pytest.mark.parametrize("label", [0, 1])
def test_diagnostics_handle_single_class_and_constant_duration(label):
    dataset = [
        (torch.tensor([2.0]), label, {
            "speaker_id": 1, "bac": 0.6 if label else 0.0, "audio_length": 5.0,
        })
    ]
    captured = {}

    def capture_figure(fig, artifact_path):
        captured[artifact_path] = fig

    with patch("utils.figure_logging.mlflow.log_figure", side_effect=capture_figure):
        run_test_evaluation(
            torch.nn.Identity(), Params(), DataLoader(dataset), torch.device("cpu"),
            eval_type="test", threshold=0.8,
        )

    duration_ax = captured["images/test_audio_length_performance_final.png"].axes[0]
    assert all(np.isnan(bar.get_height()) for bar in duration_ax.containers[0])
    bac_ax, sober_ax = captured["images/test_bac_sensitivity_final.png"].axes
    if label:
        assert np.isnan(sober_ax.patches[0].get_height())
    else:
        assert "No intoxicated samples" in bac_ax.texts[0].get_text()
    assert not plt.get_fignums()
