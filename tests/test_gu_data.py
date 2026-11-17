"""GU cache and loading checks using small synthetic recordings."""

from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from gu_data import GUData, cache_gu_data, discover_gu_audio


@pytest.fixture
def gu_root(tmp_path):
    root = tmp_path / "GU"
    # Identical basenames and source speaker IDs must survive both cohorts.
    for folder, duration in (("filtered-control", 0.5), ("filtered-intoxicated", 1.0)):
        directory = root / folder
        directory.mkdir(parents=True)
        sf.write(directory / "21_FEMALE_1003_11.wav", np.zeros(int(8000 * duration)), 8000)
    return root


def create_cache(root, cache_path):
    with patch("gu_data.opensmile.Smile") as smile:
        smile.return_value.process_file.return_value.to_numpy.side_effect = [
            np.array([[1.0, 2.0]]), np.array([[3.0, 2.0]])
        ]
        cache_gu_data(root, cache_path)


def test_cohort_paths_labels_speakers_and_durations(gu_root, tmp_path):
    cache_path = tmp_path / "features.pt"
    create_cache(gu_root, cache_path)
    data = GUData(gu_root, cache_path=cache_path, include_audio_length=True)

    assert len(data) == 2
    assert data.files == [
        "filtered-control/21_FEMALE_1003_11.wav",
        "filtered-intoxicated/21_FEMALE_1003_11.wav",
    ]
    assert data.class_labels.tolist() == [0, 1]
    assert data.speaker_ids == [2006, 2007]
    data.calculate_mu_sigma([0, 1])
    features, labels, metadata = next(iter(DataLoader(data, batch_size=2)))
    torch.testing.assert_close(features, torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
    assert labels.tolist() == [0, 1]
    assert metadata["speaker_id"].tolist() == [2006, 2007]
    assert metadata["audio_length"].tolist() == [0.5, 1.0]
    assert torch.isnan(metadata["bac"]).all()
    assert metadata["local_index"].tolist() == [-1, -1]


def test_changed_audio_is_rejected(gu_root, tmp_path):
    cache_path = tmp_path / "features.pt"
    create_cache(gu_root, cache_path)
    sf.write(gu_root / "filtered-intoxicated/21_FEMALE_1003_11.wav", np.ones(8000), 8000)
    with pytest.raises(ValueError, match="Mismatch in sha256"):
        GUData(gu_root, cache_path=cache_path)


def test_missing_cohort_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="filtered-control"):
        discover_gu_audio(tmp_path)


def test_empty_cohort_is_rejected(tmp_path):
    (tmp_path / "filtered-control").mkdir()
    with pytest.raises(ValueError, match="No WAV files"):
        discover_gu_audio(tmp_path)


def test_existing_cache_is_preserved(gu_root, tmp_path):
    cache_path = tmp_path / "features.pt"
    create_cache(gu_root, cache_path)
    original = cache_path.read_bytes()
    with pytest.raises(RuntimeError, match="already exists"):
        cache_gu_data(gu_root, cache_path)
    assert cache_path.read_bytes() == original


def test_real_opensmile_roundtrip(gu_root, tmp_path):
    cache_path = tmp_path / "real-features.pt"
    cache_gu_data(gu_root, cache_path)
    data = GUData(gu_root, cache_path=cache_path)
    data.set_mu_sigma(torch.zeros(6373), torch.ones(6373))
    for index in range(len(data)):
        features, label, _ = data[index]
        assert features.shape == (6373,)
        assert torch.isfinite(features).all()
        assert label.item() == index
