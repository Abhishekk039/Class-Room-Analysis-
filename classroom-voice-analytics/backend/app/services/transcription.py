"""
Transcription via faster-whisper. Runs fully locally/offline once the
model weights are cached on disk (first run downloads them; every run
after that needs no network access) - matching the assignment's
offline-first platform requirement.
"""
from typing import List, Optional
from dataclasses import dataclass

from faster_whisper import WhisperModel

from app.core.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE,
    WHISPER_BEAM_SIZE,
    WHISPER_CPU_THREADS,
)

_model: Optional[WhisperModel] = None


@dataclass
class RawSegment:
    start: float
    end: float
    text: str


def _get_model() -> WhisperModel:
    """Lazy-load the model once per process."""
    global _model
    if _model is None:
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=WHISPER_CPU_THREADS,
        )
    return _model


def transcribe_audio(audio_path: str) -> tuple[List[RawSegment], str]:
    """
    Transcribe a preprocessed (16kHz mono) audio file.
    Returns (segments, detected_language).

    Speed note: beam_size=1 (greedy decoding) is roughly 3-5x faster than
    beam_size=5 on CPU, at a small accuracy cost. Use 1 while iterating on
    the pipeline, bump to 5 for your final submission run.
    """
    model = _get_model()
    segments_iter, info = model.transcribe(
        audio_path,
        language=WHISPER_LANGUAGE,
        vad_filter=True,          # skip silence internally, improves accuracy and speed
        beam_size=WHISPER_BEAM_SIZE,
    )

    segments = [
        RawSegment(start=seg.start, end=seg.end, text=seg.text.strip())
        for seg in segments_iter
        if seg.text.strip()
    ]
    return segments, info.language
