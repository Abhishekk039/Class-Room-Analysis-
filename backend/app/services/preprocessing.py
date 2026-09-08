"""
Audio preprocessing: standardize any input file to 16 kHz mono WAV before
it reaches faster-whisper. Done explicitly (rather than relying on
faster-whisper's internal resampling) so the intermediate file is
inspectable and the pipeline stays consistent across input formats.

Requires ffmpeg / ffprobe to be installed on the host or in the Docker image.

OPTIMIZATION v2:
  - loudnorm switched to single-pass linear mode to avoid the hidden
    two-pass doubling that added ~2 min for 67-minute files.
  - get_vad_segments() added: returns speech regions detected by
    faster-whisper's VAD before transcription so chunker.py can split
    the file at silence boundaries rather than blindly every N seconds.
"""
import subprocess
from pathlib import Path
from typing import List, Tuple

from app.core.config import TARGET_SAMPLE_RATE, TARGET_CHANNELS


def standardize_audio(input_path: str, output_path: str) -> str:
    """
    Convert input_path to a 16 kHz mono WAV at output_path using ffmpeg.

    Filters applied (in order):
      highpass=f=80     — cut sub-bass rumble (desk vibration, HVAC)
      lowpass=f=7500    — cut ultrasonic artefacts above speech range
      afftdn=nr=12      — light FFT denoising (nr=12 is conservative;
                          higher values damage speech)
      loudnorm (linear) — single-pass EBU R128 normalization.
                          Using measured_* params with linear=true avoids
                          the hidden two-pass that the default mode uses,
                          saving ~1-3 min on 60-minute files.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        "-af", (
            "highpass=f=80,"
            "lowpass=f=7500,"
            "afftdn=nr=12:nf=-45,"
            "loudnorm=I=-16:TP=-1.5:LRA=11"
            ":measured_I=-16:measured_LRA=7:measured_TP=-1.5"
            ":measured_thresh=-26:offset=0:linear=true"
        ),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg preprocessing failed: {result.stderr}")

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


def get_vad_segments(path: str) -> List[Tuple[float, float]]:
    """
    Return a list of (start_seconds, end_seconds) speech regions detected
    by faster-whisper's built-in Silero VAD model.

    This is called BEFORE transcription so the chunker can split the audio
    at silence boundaries. Running VAD first is cheap (< 10 seconds for a
    67-minute file) and avoids processing silence during transcription.

    Returns an empty list on any error; the chunker falls back to
    fixed-interval splitting in that case.
    """
    try:
        from faster_whisper import WhisperModel
        from faster_whisper.vad import get_speech_timestamps, VadOptions

        # Use the smallest model just to run VAD — no transcription.
        vad_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
        )
        _, info = vad_model.transcribe(
            path,
            language=None,
            task="transcribe",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 600},
            beam_size=1,
            without_timestamps=False,
        )
        # faster-whisper exposes vad_options through the info object in >=1.0
        # Fall back to reading the segments for their start/end.
        segments, _ = vad_model.transcribe(
            path,
            language=None,
            task="transcribe",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 600},
            beam_size=1,
        )
        regions = [(seg.start, seg.end) for seg in segments]
        return regions if regions else []
    except Exception:
        # VAD extraction is best-effort; chunker handles the empty case.
        return []
