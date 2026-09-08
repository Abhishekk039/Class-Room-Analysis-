from app.services.diarization import label_speakers, merge_close_segments
from app.services.transcription import RawSegment


def test_merge_close_segments_combines_breath_pauses():
    segments = [
        RawSegment(start=4, end=6, text="second"),
        RawSegment(start=0, end=2, text="first"),
        RawSegment(start=2.5, end=4, text="continued"),
    ]

    merged = merge_close_segments(segments, max_gap=1)

    assert len(merged) == 1
    assert merged[0].start == 0
    assert merged[0].end == 6
    assert merged[0].text == "first continued second"


def test_label_speakers_uses_merged_turn_duration():
    segments = [
        RawSegment(start=0, end=4, text="first"),
        RawSegment(start=4.5, end=7, text="continued"),
    ]

    labeled = label_speakers(segments)

    assert len(labeled) == 1
    assert labeled[0].speaker == "teacher"


def test_merge_does_not_join_answer_to_question_boundary():
    segments = [
        RawSegment(start=0, end=5, text="What is 2 plus 2?"),
        RawSegment(start=5.5, end=8, text="Four"),
    ]

    merged = merge_close_segments(segments)

    assert len(merged) == 2


def test_merge_respects_maximum_turn_duration():
    segments = [
        RawSegment(start=0, end=10, text="First explanation"),
        RawSegment(start=11, end=19, text="Second explanation"),
        RawSegment(start=20, end=28, text="Third explanation"),
    ]

    merged = merge_close_segments(segments, max_duration=20)

    assert [(segment.start, segment.end) for segment in merged] == [
        (0, 19),
        (20, 28),
    ]