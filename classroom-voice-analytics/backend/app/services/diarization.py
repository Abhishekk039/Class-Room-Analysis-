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

IMPORTANT: Whisper's internal VAD splits a single continuous speaker into
many short chunks wherever they pause for breath (mid-sentence, between
clauses). Applying the duration threshold directly to raw Whisper segments
misreads those natural pauses as a change of speaker. We first merge
segments that are close together in time into a single "turn" - a real
turn-taking gap (someone else starts speaking) tends to be longer than a
mid-monologue breath pause.
"""
from typing import List

from app.core.config import (
    TEACHER_SEGMENT_DURATION_THRESHOLD_SECONDS,
    MERGE_GAP_THRESHOLD_SECONDS,
)
from app.services.transcription import RawSegment
from app.models.schemas import TranscriptSegment


def merge_close_segments(
    segments: List[RawSegment],
    max_gap: float = MERGE_GAP_THRESHOLD_SECONDS,
) -> List[RawSegment]:
    """
    Merge consecutive Whisper segments into one turn when the gap between
    them is small enough to be a breath pause rather than a speaker change.
    """
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        gap = seg.start - last.end
        if gap <= max_gap:
            merged[-1] = RawSegment(
                start=last.start,
                end=seg.end,
                text=f"{last.text} {seg.text}".strip(),
            )
        else:
            merged.append(seg)
    return merged


def label_speakers(segments: List[RawSegment]) -> List[TranscriptSegment]:
    turns = merge_close_segments(segments)
    labeled: List[TranscriptSegment] = []
    for seg in turns:
        duration = seg.end - seg.start
        speaker = "teacher" if duration >= TEACHER_SEGMENT_DURATION_THRESHOLD_SECONDS else "student"
        labeled.append(
            TranscriptSegment(start=seg.start, end=seg.end, text=seg.text, speaker=speaker)
        )
    return labeled
