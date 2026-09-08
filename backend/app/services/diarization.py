"""
Speaker separation heuristic.

This is a documented approximation, not real speaker-embedding diarization
(pyannote.audio would be the production choice - see docs/ARCHITECTURE.md
for the tradeoff). The brief explicitly allows heuristic/keyword-based
approaches ("approximation is acceptable").

Heuristic: teachers tend to speak in longer, continuous explanatory turns;
students tend to answer in short bursts. We flag any segment longer than
TEACHER_SEGMENT_DURATION_THRESHOLD_SECONDS as "teacher" and everything
else as "student". This is intentionally simple and its limitations are
called out explicitly in the README rather than hidden.
"""
from typing import List

from app.core.config import (
    MERGE_GAP_THRESHOLD_SECONDS,
    MAX_MERGE_DURATION_SECONDS,
    QUESTION_KEYWORDS,
    TEACHER_SEGMENT_DURATION_THRESHOLD_SECONDS,
)
from app.services.transcription import RawSegment
from app.models.schemas import TranscriptSegment


def merge_close_segments(
    segments: List[RawSegment],
    max_gap: float = MERGE_GAP_THRESHOLD_SECONDS,
    max_duration: float = MAX_MERGE_DURATION_SECONDS,
) -> List[RawSegment]:
    """Join nearby Whisper chunks that are likely one speaker turn."""
    if not segments:
        return []

    ordered = sorted(segments, key=lambda segment: (segment.start, segment.end))
    merged = [ordered[0]]
    for segment in ordered[1:]:
        previous = merged[-1]
        previous_text = previous.text.strip().lower()
        previous_is_question = previous_text.endswith("?") or any(
            previous_text.startswith(keyword) for keyword in QUESTION_KEYWORDS
        )
        combined_duration = segment.end - previous.start
        if (
            segment.start - previous.end <= max_gap
            and not previous_is_question
            and combined_duration <= max_duration
        ):
            merged[-1] = RawSegment(
                start=previous.start,
                end=max(previous.end, segment.end),
                text=f"{previous.text} {segment.text}".strip(),
            )
        else:
            merged.append(segment)
    return merged


def label_speakers(segments: List[RawSegment]) -> List[TranscriptSegment]:
    labeled: List[TranscriptSegment] = []
    for seg in merge_close_segments(segments):
        duration = seg.end - seg.start
        speaker = "teacher" if duration >= TEACHER_SEGMENT_DURATION_THRESHOLD_SECONDS else "student"
        labeled.append(
            TranscriptSegment(start=seg.start, end=seg.end, text=seg.text, speaker=speaker)
        )
    return labeled
