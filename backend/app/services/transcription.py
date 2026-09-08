"""
Transcription via faster-whisper. Runs fully locally/offline once the
model weights are cached on disk (first run downloads them; every run
after that needs no network access) — matching the offline-first requirement.

OPTIMIZATION v2 changes:
  - transcribe_chunk() added: transcribes a single ChunkSpec and offsets
    all segment timestamps by chunk.start so they are correct in the
    original recording timeline.
  - RawSegment gains a `confidence` field (avg_logprob from faster-whisper).
  - Decoding params tightened: beam_size=2, condition_on_previous_text=False,
    compression_ratio_threshold=1.8, no_speech_threshold=0.5.
  - initial_prompt loaded from the active vocabulary JSON to guide Whisper
    toward domain-specific terminology.
  - transcribe_audio() preserved unchanged for backwards compatibility and
    use in existing unit tests.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from faster_whisper import WhisperModel

from app.core.config import (
    VOCABULARY_FILE,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_SIZE,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_TASK,
    WHISPER_VAD_MIN_SILENCE_MS,
)

_model: Optional[WhisperModel] = None


@dataclass
class RawSegment:
    start: float
    end: float
    text: str
    confidence: float = 0.0          # avg_logprob mapped to [0, 1]
    suspicious: bool = False          # set by quality.py later
    correction_applied: bool = False  # set by quality.py later
    original_text: str = field(default="")

    def __post_init__(self):
        if not self.original_text:
            self.original_text = self.text


def _get_model() -> WhisperModel:
    """Lazy-load the Whisper model once per process."""
    global _model
    if _model is None:
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _model


def _load_initial_prompt() -> str:
    """Load the initial_prompt string from the active vocabulary JSON."""
    vocab_path = Path(VOCABULARY_FILE)
    if not vocab_path.exists():
        # Try relative to the backend directory.
        alt = Path(__file__).parent.parent.parent / VOCABULARY_FILE
        vocab_path = alt if alt.exists() else None

    if vocab_path and vocab_path.exists():
        try:
            data = json.loads(vocab_path.read_text(encoding="utf-8"))
            return data.get("initial_prompt", "")
        except Exception:
            pass
    return ""


def _logprob_to_confidence(avg_logprob: float) -> float:
    """
    Convert faster-whisper avg_logprob (typically in [-1, 0]) to a
    [0, 1] confidence score. Values below -1 are clamped to 0.
    """
    import math
    # avg_logprob is log-probability; exp maps it to probability.
    return round(min(1.0, max(0.0, math.exp(avg_logprob))), 3)


def _decode_params() -> dict:
    return dict(
        language=WHISPER_LANGUAGE,
        task=WHISPER_TASK,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": WHISPER_VAD_MIN_SILENCE_MS},
        beam_size=WHISPER_BEAM_SIZE,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
        compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
        condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        initial_prompt=_load_initial_prompt() or None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str) -> Tuple[List[RawSegment], str]:
    """
    Transcribe a preprocessed (16 kHz mono) audio file.
    Returns (segments, detected_language).

    Preserved for backwards compatibility and existing unit tests.
    For long recordings, use transcribe_chunk() via pipeline.py instead.
    """
    model = _get_model()
    segments_iter, info = model.transcribe(audio_path, **_decode_params())

    segments = [
        RawSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            # getattr with default so test mocks without avg_logprob still work.
            confidence=_logprob_to_confidence(getattr(seg, "avg_logprob", -0.5)),
        )
        for seg in segments_iter
        if seg.text.strip() and seg.end > seg.start
    ]
    segments.sort(key=lambda s: (s.start, s.end))
    return segments, info.language


def transcribe_chunk(chunk) -> List[RawSegment]:
    """
    Transcribe a single ChunkSpec and return RawSegments with timestamps
    adjusted to the original recording timeline.

    `chunk` is a ChunkSpec from services/chunker.py.
    The overlap region timestamps are already included in chunk.start/end;
    quality.py will remove duplicates that appear at chunk boundaries.
    """
    model = _get_model()
    params = _decode_params()
    # Per-chunk decoding: never carry context across chunk boundaries.
    params["condition_on_previous_text"] = False

    segments_iter, _ = model.transcribe(chunk.path, **params)

    # The slice file starts at (chunk.start - overlap), so we need to
    # calculate the offset between the slice file's t=0 and the original
    # recording's timeline.
    #
    # chunk.start is the VAD region start (without overlap).
    # The actual slice file starts at max(0, chunk.start - CHUNK_OVERLAP_SECONDS).
    from app.core.config import CHUNK_OVERLAP_SECONDS
    import math as _math
    slice_offset = max(0.0, chunk.start - CHUNK_OVERLAP_SECONDS)

    results: List[RawSegment] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text or seg.end <= seg.start:
            continue
        results.append(RawSegment(
            start=round(slice_offset + seg.start, 3),
            end=round(slice_offset + seg.end, 3),
            text=text,
            confidence=_logprob_to_confidence(seg.avg_logprob),
        ))

    results.sort(key=lambda s: (s.start, s.end))
    return results
