from app.models.schemas import TranscriptSegment
from app.services.transcription import transcribe_audio
from app.services.analysis import (
    detect_questions,
    count_student_responses,
    compute_talk_time,
    compute_silence,
)


class FakeWhisperSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class FakeWhisperInfo:
    language = "en"


class FakeWhisperModel:
    def transcribe(self, audio_path, **kwargs):
        return iter([
            FakeWhisperSegment(8, 10, " later "),
            FakeWhisperSegment(4, 4, "invalid"),
            FakeWhisperSegment(2, 4, " earlier "),
            FakeWhisperSegment(10, 12, "   "),
        ]), FakeWhisperInfo()


def make_segment(start, end, text, speaker):
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker)


def test_transcribe_audio_filters_invalid_segments_and_sorts(monkeypatch):
    monkeypatch.setattr("app.services.transcription._get_model", lambda: FakeWhisperModel())

    segments, language = transcribe_audio("recording.wav")

    assert language == "en"
    assert [(segment.start, segment.end, segment.text) for segment in segments] == [
        (2, 4, "earlier"),
        (8, 10, "later"),
    ]


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


def test_count_student_responses_rejects_boundary_question():
    segments = [
        make_segment(0, 5, "What is 2 plus 2?", "teacher"),
        make_segment(5, 7, "What is the answer?", "student"),
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
