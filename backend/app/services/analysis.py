"""
Basic classroom analysis: question detection, response counting,
talk-time totals, and silence duration - built on top of the
speaker-labeled transcript from diarization.py.
"""
from typing import List

from app.core.config import QUESTION_KEYWORDS, RESPONSE_WINDOW_SECONDS
from app.models.schemas import TranscriptSegment


def detect_questions(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """
    Mark teacher segments as questions using two signals combined:
    ends with '?' (Whisper often restores punctuation), OR starts with
    a question keyword/phrase. Combining both catches cases where
    punctuation restoration misses the mark.
    """
    for seg in segments:
        if seg.speaker != "teacher":
            continue
        text_lower = seg.text.strip().lower()
        ends_with_question_mark = text_lower.endswith("?")
        starts_with_keyword = any(text_lower.startswith(k) for k in QUESTION_KEYWORDS)
        seg.is_question = ends_with_question_mark or starts_with_keyword
    return segments


def count_student_responses(segments: List[TranscriptSegment]) -> int:
    """
    A student segment counts as a "response" if it starts within
    RESPONSE_WINDOW_SECONDS of a teacher question segment ending.
    """
    response_count = 0
    question_end_times = [s.end for s in segments if s.speaker == "teacher" and s.is_question]

    for seg in segments:
        if seg.speaker != "student":
            continue
        text_lower = seg.text.strip().lower()
        looks_like_question = text_lower.endswith("?") or any(
            text_lower.startswith(keyword) for keyword in QUESTION_KEYWORDS
        )
        if looks_like_question:
            continue
        for q_end in question_end_times:
            if q_end < seg.start <= q_end + RESPONSE_WINDOW_SECONDS:
                response_count += 1
                break
    return response_count


def compute_talk_time(segments: List[TranscriptSegment]) -> tuple[float, float]:
    """Returns (teacher_talk_time_seconds, student_talk_time_seconds)."""
    teacher_time = sum(s.end - s.start for s in segments if s.speaker == "teacher")
    student_time = sum(s.end - s.start for s in segments if s.speaker == "student")
    return teacher_time, student_time


def compute_silence(segments: List[TranscriptSegment], total_duration: float) -> float:
    """Total duration minus all speech segment durations, floored at 0."""
    total_speech = sum(s.end - s.start for s in segments)
    return max(0.0, total_duration - total_speech)
