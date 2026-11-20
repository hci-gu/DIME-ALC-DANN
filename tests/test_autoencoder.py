"""Check the ALC evaluation protocol without training a neural network."""

import csv
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import alc_data
import autoencoder
from alc_data import ALCData
from autoencoder import (
    SpeakerScores,
    SpeakerSplit,
    _aggregate_metrics,
    _diagnostic_rows,
    _fit_pca,
    _macro_roc_recall_at_fpr,
    _reconstruction_errors,
    _score_errors,
    _speaker_features,
    _speaker_partitions,
    _speaker_splits,
    _split_sober_indices,
    _threshold_sweep,
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


def test_sweep_selects_highest_recall_within_development_budget():
    splits = {6: SpeakerSplit([], [], [0, 1, 2, 3], [4, 5, 6, 7])}
    calibration = torch.arange(5, dtype=torch.float32)
    scores = {6: SpeakerScores(calibration, torch.tensor([1, 3.3, 3.55, 3.75, 3.35, 3.55, 3.7, 3.9]))}

    rows, selected = _threshold_sweep(splits, scores, target_fpr=0.25)

    assert [row["pooled_sober_fpr"] for row in rows] == [0.75, 0.5, 0.25, 0]
    assert [row["pooled_recall"] for row in rows] == [1, 0.75, 0.5, 0.25]
    assert selected == 0.9
    always_high = {6: SpeakerScores(calibration, torch.full((8,), 5.0))}
    assert _threshold_sweep(splits, always_high, target_fpr=0.25)[1] is None
    assert _score_errors(calibration, always_high[6].test, 4, quantile=None)["recall"] == 0


def test_speaker_partitions_are_disjoint_and_stable():
    speakers = list(range(162))
    partitions = _speaker_partitions(speakers, seed=1999)

    assert [len(group) for group in partitions.values()] == [97, 32, 33]
    assert sorted(sum(partitions.values(), [])) == speakers
    assert _speaker_partitions(speakers[::-1], seed=1999) == partitions


def test_pca_scores_centered_reconstruction_residual():
    features = torch.tensor([[2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
    model = _fit_pca(features, n_components=8)

    errors = _reconstruction_errors(model, torch.tensor([[5.0, 1.0], [3.0, 3.0]]))

    torch.testing.assert_close(errors, torch.tensor([0.0, 2.0]))


def test_roc_interpolates_the_same_fpr_for_each_speaker():
    splits = {speaker_id: SpeakerSplit([], [], [0, 1], [2, 3]) for speaker_id in (6, 7)}
    scores = {
        6: SpeakerScores(torch.empty(0), torch.tensor([0.0, 1.0, 2.0, 3.0])),
        7: SpeakerScores(torch.empty(0), torch.ones(4)),
    }

    assert _macro_roc_recall_at_fpr(splits, scores, target_fpr=0.2) == pytest.approx(0.6)


def test_prompt_diagnostic_pairs_only_annotated_test_utterances():
    data = SimpleNamespace(
        files=["0062000001_h_01.wav", "0062000002_h_01.wav", "0061000001_h_00.wav", "0061000002_h_00.wav"],
        class_labels=torch.tensor([0, 0, 1, 1]),
        paired_utterance_ids=[None, None, "0062000001", "0062000002"],
    )
    splits = {6: SpeakerSplit([], [1], [0], [2, 3])}
    scores = {6: SpeakerScores(torch.tensor([2.0]), torch.tensor([1.0, 4.0, 5.0]))}

    utterances, pairs = _diagnostic_rows(data, splits, {"held_out": scores})

    assert [row["split"] for row in utterances] == ["calibration", "test", "test", "test"]
    assert len(pairs) == 1
    assert pairs[0]["sober_file"] == data.files[0]
    assert pairs[0]["intoxicated_file"] == data.files[2]
    assert pairs[0]["error_difference"] == 3.0


def test_missing_egemaps_cache_reports_build_command(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError) as exc:
        ALCData(feature_set="eGeMAPSv02")

    assert str(tmp_path / ".cache" / "alc-egemaps-features.pt") in str(exc.value)
    assert "uv run --no-sync python alc_data.py --feature-set eGeMAPSv02" in str(exc.value)


def test_egemaps_cache_round_trip_keeps_compare_cache_separate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    audio_dir = tmp_path / "wav" / "h"
    labels_dir = tmp_path / "labels" / "h"
    audio_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    utterance_id = "0061000001"
    counterpart_id = "0062000001"
    stem = f"{utterance_id}_h_00"
    (audio_dir / f"{stem}.wav").write_bytes(b"synthetic wav")
    annotation = {
        "utterance": stem, "utt": utterance_id, "spn": "006", "o_utt": counterpart_id,
        "item": "001", "o_item": "001", "alc": "a", "bak": "0.001",
    }
    labels = [{"name": name, "value": value} for name, value in annotation.items()]
    (labels_dir / f"{stem}_annot.json").write_text(
        json.dumps({"levels": [{"items": [{"labels": labels}]}]}), encoding="utf-8",
    )
    compare_cache = tmp_path / ".cache" / "alc-opensmile-features.pt"
    compare_cache.parent.mkdir()
    compare_cache.write_bytes(b"existing cache")
    processor = Mock()
    processor.process_file.return_value.to_numpy.return_value = [[1.0] * 88]
    smile = Mock(return_value=processor)
    monkeypatch.setattr(alc_data.opensmile, "Smile", smile)

    alc_data.cache_alc_data(audio_dir, feature_set_name="eGeMAPSv02")
    data = ALCData(data_path=tmp_path, feature_set="eGeMAPSv02")

    assert smile.call_args.kwargs["feature_set"] == alc_data.opensmile.FeatureSet.eGeMAPSv02
    assert compare_cache.read_bytes() == b"existing cache"
    assert data.tensors_dict[f"{stem}.wav"].shape == (88,)
    assert data.paired_utterance_ids == [counterpart_id]


def test_shared_cli_fits_only_reference_speakers_and_reports_held_out_metrics(monkeypatch):
    speakers = list(range(6, 12))
    data = ALCData.__new__(ALCData)
    data.speaker_ids = [speaker_id for speaker_id in speakers for _ in range(6)]
    data.files = [f"{speaker_id:03d}{index:07d}_h.wav" for index, speaker_id in enumerate(data.speaker_ids)]
    data.tensors_dict = {file: torch.tensor([float(index % 6), 1.0]) for index, file in enumerate(data.files)}
    data.class_labels = torch.tensor(([0] * 5 + [1]) * len(speakers))
    data.class_mapping = {"na": 0, "a": 1}
    data.cache_dict = {"feature_set": "eGeMAPSv02", "feature_level": "Functionals"}
    data.bac_values = data.class_labels.tolist()
    data.train_speaker_mapping = {}
    data.include_audio_length = False
    data.transforms = None
    data.verbose = False
    data.paired_utterance_ids = [None] * len(data.files)
    data.calculate_mu_sigma = Mock(wraps=data.calculate_mu_sigma)

    load = Mock(return_value=data)
    fit = Mock(return_value=torch.nn.Identity())
    tracking = Mock()
    tracking.start_run.return_value = nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="synthetic")))
    monkeypatch.setattr(alc_data, "ALCData", load)
    monkeypatch.setattr(autoencoder, "_fit_autoencoder", fit)
    monkeypatch.setitem(sys.modules, "mlflow", tracking)
    monkeypatch.setattr(sys, "argv", [
        "autoencoder.py", "--threshold-sweep", "--method", "shared-mlp",
        "--feature-set", "eGeMAPSv02", "--device", "cpu",
    ])

    autoencoder.main()

    partitions = _speaker_partitions(speakers, seed=1999)
    fit.assert_called_once()
    training_calls = data.calculate_mu_sigma.call_args_list[:len(partitions["shared_training"])]
    assert [{data.speaker_ids[index] for index in call.args[0]} for call in training_calls] == [
        {speaker_id} for speaker_id in partitions["shared_training"]
    ]
    assert all(data.class_labels[index] == 0 for call in training_calls for index in call.args[0])
    load.assert_called_once_with(seed=1999, feature_set="eGeMAPSv02")
    tracking.set_experiment.assert_called_once_with("autoencoder")
    tracking.log_dict.assert_any_call(partitions, "speaker_partitions.json")
    artifacts = {call.args[1]: list(csv.DictReader(call.args[0].splitlines())) for call in tracking.log_text.call_args_list}
    assert len(artifacts["threshold_sweep.csv"]) == 4
    assert [int(row["speaker_id"]) for row in artifacts["speaker_metrics.csv"]] == partitions["held_out"]
    assert {row["cohort"] for row in artifacts["utterance_scores.csv"]} == {"development", "held_out"}


def test_logistic_cli_fits_each_speaker_and_reserves_a_common_test_set(monkeypatch):
    speakers = list(range(6, 12))
    data = ALCData.__new__(ALCData)
    data.speaker_ids = [speaker_id for speaker_id in speakers for _ in range(90)]
    data.files = [f"{speaker_id:03d}{index:07d}_h.wav" for index, speaker_id in enumerate(data.speaker_ids)]
    values = torch.cat([torch.linspace(-1, 1, 60), torch.linspace(10, 11, 30)])
    data.tensors_dict = {
        file: (values[index % 90] * (-1) ** speaker_id).reshape(1)
        for index, (file, speaker_id) in enumerate(zip(data.files, data.speaker_ids, strict=True))
    }
    data.class_labels = torch.tensor(([0] * 60 + [1] * 30) * len(speakers))
    data.class_mapping = {"na": 0, "a": 1}
    data.cache_dict = {"feature_set": "eGeMAPSv02", "feature_level": "Functionals"}
    data.bac_values = data.class_labels.tolist()
    data.train_speaker_mapping = {}
    data.include_audio_length = False
    data.transforms = None
    data.verbose = False

    baseline_splits, _ = _speaker_splits(data, seed=1999)
    common_splits, skipped = _speaker_splits(data, seed=1999, reserve_intoxicated_training=True)
    assert not skipped
    for speaker_id, split in common_splits.items():
        assert (len(split.train), len(split.intoxicated_enrollment), len(split.test)) == (36, 18, 24)
        assert set(split.intoxicated_enrollment).isdisjoint(split.test)
        assert set(split.intoxicated_enrollment + split.intoxicated) == set(baseline_splits[speaker_id].intoxicated)
        assert split.sober_test == baseline_splits[speaker_id].sober_test

    fit = Mock(wraps=autoencoder._fit_logistic_regression)
    tracking = Mock()
    tracking.start_run.return_value = nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="synthetic")))
    monkeypatch.setattr(alc_data, "ALCData", Mock(return_value=data))
    monkeypatch.setattr(autoencoder, "_fit_logistic_regression", fit)
    monkeypatch.setitem(sys.modules, "mlflow", tracking)
    monkeypatch.setattr(sys, "argv", [
        "autoencoder.py", "--method", "logistic", "--feature-set", "eGeMAPSv02", "--threshold-sweep",
    ])

    autoencoder.main()

    partitions = _speaker_partitions(speakers, seed=1999)
    scored_speakers = partitions["development"] + partitions["held_out"]
    assert fit.call_count == len(scored_speakers)
    for speaker_id, call in zip(scored_speakers, fit.call_args_list, strict=True):
        split = common_splits[speaker_id]
        sober = torch.stack([data.tensors_dict[data.files[index]] for index in split.train])
        intoxicated = torch.stack([data.tensors_dict[data.files[index]] for index in split.intoxicated_enrollment])
        torch.testing.assert_close(call.args[1], (intoxicated - sober.mean(dim=0)) / sober.std(dim=0, unbiased=False))

    artifacts = {call.args[1]: list(csv.DictReader(call.args[0].splitlines())) for call in tracking.log_text.call_args_list}
    held_out_rows = [row for row in artifacts["utterance_scores.csv"] if row["cohort"] == "held_out" and row["split"] == "test"]
    assert {row["file"] for row in held_out_rows} == {
        data.files[index] for speaker_id in partitions["held_out"] for index in common_splits[speaker_id].test
    }
    assert "decision_score" in held_out_rows[0]
    assert tracking.log_metrics.call_args.args[0]["macro_roc_recall_at_target_fpr"] == pytest.approx(1.0)
