from app.models.schemas import TranscriptSegment
from app.services.analysis import (
    detect_questions,
    count_student_responses,
    compute_talk_time,
    compute_silence,
)


def make_segment(start, end, text, speaker):
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker)


def test_detect_questions_punctuation():
    segments = [make_segment(0, 5, "What fraction is left?", "teacher")]
    result = detect_questions(segments)
    assert result[0].is_question is True


def test_detect_questions_keyword_without_mark():
    segments = [make_segment(0, 5, "Can you explain that", "teacher")]
    result = detect_questions(segments)
    assert result[0].is_question is True


def test_detect_questions_ignores_students():
    segments = [make_segment(0, 5, "What is this?", "student")]
    result = detect_questions(segments)
    assert result[0].is_question is False


def test_count_student_responses_within_window():
    segments = [
        make_segment(0, 5, "What is 2 plus 2?", "teacher"),
        make_segment(6, 8, "Four", "student"),
    ]
    segments = detect_questions(segments)
    assert count_student_responses(segments) == 1


def test_count_student_responses_outside_window():
    segments = [
        make_segment(0, 5, "What is 2 plus 2?", "teacher"),
        make_segment(20, 22, "Four", "student"),
    ]
    segments = detect_questions(segments)
    assert count_student_responses(segments) == 0


def test_compute_talk_time():
    segments = [
        make_segment(0, 10, "explain", "teacher"),
        make_segment(10, 15, "answer", "student"),
    ]
    teacher_time, student_time = compute_talk_time(segments)
    assert teacher_time == 10
    assert student_time == 5


def test_compute_silence():
    segments = [make_segment(0, 10, "explain", "teacher")]
    silence = compute_silence(segments, total_duration=15)
    assert silence == 5
