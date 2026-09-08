"""
Audio preprocessing: standardize any input file to 16kHz mono WAV before
it reaches faster-whisper. Done explicitly (rather than relying on
faster-whisper's internal resampling) so the intermediate file is
inspectable and the pipeline stays consistent across input formats.

Requires ffmpeg to be installed on the host / in the Docker image.
"""
import subprocess
from pathlib import Path

from app.core.config import (
    TARGET_SAMPLE_RATE,
    TARGET_CHANNELS,
    HIGHPASS_FREQUENCY_HZ,
)


def standardize_audio(input_path: str, output_path: str) -> str:
    """
    Convert input_path to a 16kHz mono WAV at output_path using ffmpeg.

    Filter chain, in order:
    1. highpass - cuts low-frequency rumble (fans, desk/mic handling noise,
       distant traffic) below the speech range, which otherwise confuses
       both loudness normalization and Whisper's VAD.
    2. loudnorm - evens out mic-distance/volume variation typical of a
       single classroom mic picking up speakers at different distances.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    audio_filters = f"highpass=f={HIGHPASS_FREQUENCY_HZ},loudnorm"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        "-af", audio_filters,
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg preprocessing failed: {result.stderr}")

    return output_path


def trim_audio_for_dev(input_path: str, output_path: str, seconds: int = 300) -> str:
    """
    Dev-only helper: cut the first `seconds` of an audio file so you can
    iterate on the pipeline in seconds instead of minutes. Not part of the
    production pipeline - call this manually while testing, e.g.:

        python -c "from app.services.preprocessing import trim_audio_for_dev; \\
            trim_audio_for_dev('storage/uploads/full.wav', 'storage/uploads/sample_5min.wav')"

    Then upload sample_5min.wav instead of the full recording while you're
    debugging diarization thresholds or question detection.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", input_path, "-t", str(seconds), "-c", "copy", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {result.stderr}")
    return output_path


def get_audio_duration(path: str) -> float:
    """Return duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())
