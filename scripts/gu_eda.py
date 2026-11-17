"""Read filename metadata and audio diagnostics for the two filtered GU folders."""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


FOLDERS = {"filtered-control": "na", "filtered-intoxicated": "a"}
FILENAME_PATTERN = re.compile(
    r"(?P<age>\d+)_(?P<sex>[A-Za-z]+)_(?P<speaker_id>\d+)_(?P<utterance_id>\d+)"
)


def parse_filename(path: Path) -> dict:
    """Preserve raw age and IDs; treat the age-zero sentinel as unknown."""
    match = FILENAME_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unexpected GU filename: {path.name}")
    values = match.groupdict()
    age_raw = int(values.pop("age"))
    return {**values, "sex": values["sex"].title(), "age_raw": age_raw,
            "age": age_raw if age_raw > 0 else None}


def audio_diagnostics(path: Path) -> dict:
    """Measure all channels in blocks, with a fixed -40 dBFS quiet threshold.

    Quiet samples are a level diagnostic, not a speech/silence classifier.
    The digest covers decoded float32 audio plus rate and channel count.
    """
    with sf.SoundFile(path) as audio:
        if audio.frames == 0:
            raise ValueError(f"Empty recording: {path}")
        frames, rate, channels, subtype = (
            audio.frames, audio.samplerate, audio.channels, audio.subtype
        )
        digest = hashlib.sha256(f"{rate}:{channels}:".encode())
        total = quiet = clipped = 0
        sum_squares = peak = 0.0
        for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError(f"Non-finite audio samples: {path}")
            magnitude = np.abs(block)
            total += block.size
            quiet += int((magnitude < 0.01).sum())
            clipped += int((magnitude >= 0.999).sum())
            sum_squares += float(np.square(block.astype(np.float64)).sum())
            peak = max(peak, float(magnitude.max()))
            digest.update(block.astype("<f4", copy=False).tobytes())
    rms = np.sqrt(sum_squares / total)
    return {
        "sound_duration": frames / rate, "sample_rate": rate,
        "channels": channels, "subtype": subtype, "frames": frames,
        "rms_dbfs": float(20 * np.log10(max(rms, 1e-12))),
        "peak": peak, "quiet_fraction": quiet / total,
        "clipping_fraction": clipped / total, "audio_sha256": digest.hexdigest(),
    }


def load_gu_data(root: Path) -> pd.DataFrame:
    """Read only WAV files in the explicitly selected filtered directories.

    Fail on malformed or unreadable files instead of silently dropping them.
    Numeric IDs are local to each cohort. Prefix with C (control) or I
    (intoxicated) to identify distinct people across the two cohorts.
    """
    rows = []
    for folder, label in FOLDERS.items():
        directory = root / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
        if not files:
            raise ValueError(f"No WAV files found in {directory}")
        for path in files:
            metadata = parse_filename(path)
            source_speaker_id = metadata["speaker_id"]
            metadata["speaker_id"] = f"{'C' if label == 'na' else 'I'}{source_speaker_id}"
            rows.append({
                "wav_file": path.name, "relative_path": path.relative_to(root).as_posix(),
                "class_label": label, "class_name": "Control" if label == "na" else "Intoxicated",
                **metadata,
                "source_speaker_id": source_speaker_id,
                "recording_group": metadata["speaker_id"],
                **audio_diagnostics(path),
            })
    result = pd.DataFrame(rows)
    result["age"] = result["age"].astype("Int64")
    return result
