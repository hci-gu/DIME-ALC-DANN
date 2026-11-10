"""Run a saved DANN classifier on consecutive microphone windows."""

import argparse
import queue
import sys
from pathlib import Path

import numpy as np
import opensmile
import sounddevice as sd
import torch

from model import DANN
from params import Params


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not np.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive duration")
    return seconds


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Path to a DANN .pth checkpoint")
    parser.add_argument("--window-seconds", type=positive_seconds, default=10.0)
    parser.add_argument("--input-device", help="Microphone index or name (default: system input)")
    parser.add_argument("--sample-rate", type=positive_integer, help="Default: microphone native rate")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args(argv)
    if not args.list_devices and args.checkpoint is None:
        parser.error("--checkpoint is required unless --list-devices is used")
    if args.input_device is not None:
        try:
            args.input_device = int(args.input_device)
        except ValueError:
            pass
    return args


def load_checkpoint(path: Path) -> tuple[DANN, torch.Tensor, torch.Tensor]:
    """Load a repository DANN checkpoint and its training normalization on CPU."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        print(checkpoint["threshold"])
        print(checkpoint["bac_limit"])
        p = Params(**checkpoint["params"])
        p.device = "cpu"
        if p.extractor_input_dimension != 6373 or p.classifier_output_dimension != 1:
            raise ValueError("expected 6373 ComParE features and one classifier output")
        mu = torch.as_tensor(checkpoint["mu"], dtype=torch.float32)
        sigma = torch.as_tensor(checkpoint["sigma"], dtype=torch.float32)
        if mu.shape != (6373,) or sigma.shape != mu.shape:
            raise ValueError("mu and sigma must each have shape (6373,)")
        if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all():
            raise ValueError("normalization statistics must be finite")
        if (sigma < 0).any():
            raise ValueError("sigma must be nonnegative")
        model = DANN(p)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        return model, mu, sigma
    except Exception as error:
        raise RuntimeError(f"Could not load compatible DANN checkpoint {path}: {error}") from error


class WindowCapture:
    """Assemble exact windows in the audio callback without waiting for inference."""

    def __init__(self, window_samples: int):
        if window_samples < 1:
            raise ValueError("window duration must contain at least one sample")
        self.windows: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
        self.buffer = np.empty(window_samples, dtype=np.float32)
        self.position = 0
        self.error: str | None = None

    def callback(self, indata: np.ndarray, frames: int, time: object, status: object) -> None:
        """Copy mono samples only; feature extraction runs in the main thread."""
        try:
            if status:
                raise RuntimeError(f"Microphone capture failed: {status}")
            offset = 0
            while offset < frames:
                count = min(frames - offset, self.buffer.size - self.position)
                self.buffer[self.position:self.position + count] = indata[offset:offset + count, 0]
                offset += count
                self.position += count
                if self.position == self.buffer.size:
                    try:
                        self.windows.put_nowait(self.buffer)
                    except queue.Full as error:
                        raise RuntimeError(
                            "Audio queue full: inference cannot keep up with capture. "
                            "Try a longer --window-seconds duration."
                        ) from error
                    self.buffer = np.empty_like(self.buffer)
                    self.position = 0
        except Exception as error:
            self.error = str(error)
            raise sd.CallbackAbort from error

    def check_error(self) -> None:
        if self.error is not None:
            raise RuntimeError(self.error)


def predict_window(
    audio: np.ndarray,
    sample_rate: int,
    processor: opensmile.Smile,
    model: DANN,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[float, float]:
    """Return intoxication probability and raw digital RMS for one mono window."""
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio must be a nonempty, finite mono window")
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    features = torch.tensor(
        processor.process_signal(audio, sample_rate).to_numpy(), dtype=torch.float32
    )
    if features.shape != (1, mu.numel()) or not torch.isfinite(features).all():
        raise ValueError("OpenSMILE returned unexpected dimensions or nonfinite features")
    x = features.squeeze(0)
    x = torch.where(sigma > 0, (x - mu) / sigma, torch.zeros_like(x))
    if not torch.isfinite(x).all():
        raise ValueError("Normalized features contain NaN or infinite values")
    with torch.inference_mode():
        probability = model.predict(x.unsqueeze(0), return_type="probability")
    if probability.numel() != 1 or not torch.isfinite(probability).all():
        raise ValueError("Model returned an invalid probability")
    value = probability.item()
    if not 0 <= value <= 1:
        raise ValueError("Model probability is outside [0, 1]")
    return value, rms


def run_microphone(
    args: argparse.Namespace,
    model: DANN,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> None:
    """Capture continuously until Ctrl+C or a capture/inference failure."""
    device = sd.query_devices(args.input_device, "input")
    sample_rate = args.sample_rate or int(device["default_samplerate"])
    sd.check_input_settings(
        device=args.input_device, channels=1, dtype="float32", samplerate=sample_rate
    )
    capture = WindowCapture(round(args.window_seconds * sample_rate))
    processor = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    print(
        f"Listening: {device['name']} | {sample_rate} Hz | "
        f"{capture.buffer.size / sample_rate:g}s windows | CPU | Ctrl+C to stop",
        flush=True,
    )
    with sd.InputStream(
        device=args.input_device, samplerate=sample_rate,
        channels=1, dtype="float32", callback=capture.callback,
    ) as stream:
        index = 0
        while True:
            capture.check_error()
            if not stream.active:
                raise RuntimeError("Microphone stream stopped unexpectedly")
            try:
                audio = capture.windows.get(timeout=0.1)
            except queue.Empty:
                continue
            capture.check_error()
            probability, rms = predict_window(audio, sample_rate, processor, model, mu, sigma)
            capture.check_error()
            index += 1
            print(f"window={index} p(intoxicated)={probability:.4f} rms={rms:.6f}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.list_devices:
            print(sd.query_devices())
            return 0
        model, mu, sigma = load_checkpoint(args.checkpoint)
        run_microphone(args, model, mu, sigma)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
