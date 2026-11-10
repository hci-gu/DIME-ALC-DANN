"""Inference checks using synthetic audio and mocked microphone capture."""

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import numpy as np
import opensmile
import pytest
import sounddevice as sd
import torch

import inference
from model import DANN
from params import Params


@pytest.fixture
def checkpoint_path(tmp_path):
    p = Params(extractor_n_layers=1, classifier_n_layers=1, discriminator_n_layers=1)
    model = DANN(p)
    mu = torch.ones(6373)
    sigma = torch.full((6373,), 2.0)
    sigma[0] = 0
    path = tmp_path / "synthetic.pth"
    torch.save({
        "params": asdict(p), "model_state_dict": model.state_dict(),
        "mu": mu, "sigma": sigma,
    }, path)
    return path


def test_window_boundaries_and_leftover():
    capture = inference.WindowCapture(5)
    samples = np.arange(13, dtype=np.float32).reshape(-1, 1)
    for start, stop in [(0, 2), (2, 9), (9, 13)]:
        capture.callback(samples[start:stop], stop - start, None, None)
    np.testing.assert_array_equal(capture.windows.get_nowait(), np.arange(5))
    np.testing.assert_array_equal(capture.windows.get_nowait(), np.arange(5, 10))
    np.testing.assert_array_equal(capture.buffer[:capture.position], np.arange(10, 13))


def test_queue_overflow_is_reported():
    capture = inference.WindowCapture(2)
    with pytest.raises(sd.CallbackAbort):
        capture.callback(np.zeros((8, 1), dtype=np.float32), 8, None, None)
    assert capture.windows.qsize() == 3
    with pytest.raises(RuntimeError, match="queue full"):
        capture.check_error()


def test_audio_overflow_is_reported():
    capture = inference.WindowCapture(2)
    status = sd.CallbackFlags()
    status.input_overflow = True
    with pytest.raises(sd.CallbackAbort):
        capture.callback(np.zeros((2, 1), dtype=np.float32), 2, None, status)
    with pytest.raises(RuntimeError, match="overflow"):
        capture.check_error()


def test_real_features_normalization_probability_and_rms(checkpoint_path):
    model, mu, sigma = inference.load_checkpoint(checkpoint_path)
    assert not model.training
    assert next(model.parameters()).device.type == "cpu"
    processor = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    rate = 16000
    audio = (0.1 * np.sin(2 * np.pi * 200 * np.arange(rate) / rate)).astype(np.float32)
    raw = torch.tensor(processor.process_signal(audio, rate).to_numpy(), dtype=torch.float32)
    expected = (raw - mu) / sigma
    expected[:, 0] = 0
    with patch.object(model, "predict", wraps=model.predict) as predict:
        probability, rms = inference.predict_window(audio, rate, processor, model, mu, sigma)
        torch.testing.assert_close(predict.call_args.args[0], expected)
    assert 0 <= probability <= 1
    assert rms == pytest.approx(0.1 / np.sqrt(2))
    probability, rms = inference.predict_window(np.zeros(rate, dtype=np.float32), rate, processor, model, mu, sigma)
    assert 0 <= probability <= 1
    assert rms == 0


@pytest.mark.parametrize("kind", ["missing-key", "bad-shape", "nonfinite", "negative", "bad-weights"])
def test_incompatible_checkpoint(checkpoint_path, kind):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    if kind == "missing-key":
        del checkpoint["mu"]
    elif kind == "bad-shape":
        checkpoint["sigma"] = torch.ones(2)
    elif kind == "nonfinite":
        checkpoint["mu"][0] = float("nan")
    elif kind == "negative":
        checkpoint["sigma"][0] = -1
    else:
        checkpoint["model_state_dict"] = {}
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError, match="compatible DANN checkpoint"):
        inference.load_checkpoint(checkpoint_path)


@pytest.mark.parametrize("args", [
    [], ["--checkpoint", "x", "--window-seconds", "0"],
    ["--checkpoint", "x", "--window-seconds", "nan"],
    ["--checkpoint", "x", "--sample-rate", "-1"],
])
def test_invalid_cli(args):
    with pytest.raises(SystemExit) as error:
        inference.parse_args(args)
    assert error.value.code == 2


def test_cli_help_and_listing(capsys):
    with pytest.raises(SystemExit) as error:
        inference.parse_args(["--help"])
    assert error.value.code == 0
    with patch.object(sd, "query_devices", return_value="mock microphone"):
        assert inference.main(["--list-devices"]) == 0
    assert "mock microphone" in capsys.readouterr().out


def test_missing_checkpoint(tmp_path, capsys):
    assert inference.main(["--checkpoint", str(tmp_path / "missing.pth")]) == 1
    assert "Could not load compatible DANN checkpoint" in capsys.readouterr().err


def test_ctrl_c_closes_capture(checkpoint_path, capsys):
    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.active = True

    def start_stream(**kwargs):
        audio = np.zeros((160, 1), dtype=np.float32)
        kwargs["callback"](audio, len(audio), None, None)
        return stream

    with (
        patch.object(sd, "query_devices", return_value={"name": "mock", "default_samplerate": 16000}),
        patch.object(sd, "check_input_settings"),
        patch.object(sd, "InputStream", side_effect=start_stream),
        patch.object(inference, "predict_window", side_effect=[(0.5, 0.0), KeyboardInterrupt]),
    ):
        assert inference.main([
            "--checkpoint", str(checkpoint_path), "--window-seconds", "0.005",
        ]) == 0
    assert stream.__exit__.call_args.args[0] is KeyboardInterrupt
    output = capsys.readouterr().out
    assert "window=1 p(intoxicated)=0.5000 rms=0.000000" in output
    assert "window=2" not in output
    assert "Stopped." in output
