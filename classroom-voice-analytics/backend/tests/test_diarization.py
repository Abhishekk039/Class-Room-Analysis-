from app.services.transcription import RawSegment
from app.services.diarization import merge_close_segments, label_speakers


def test_merge_close_segments_merges_small_gaps():
    segments = [
        RawSegment(start=0, end=5, text="Let me show you"),
        RawSegment(start=5.5, end=8, text="this is the kit"),  # 0.5s gap, should merge
    ]
    merged = merge_close_segments(segments, max_gap=2.0)
    assert len(merged) == 1
    assert merged[0].start == 0
    assert merged[0].end == 8
    assert merged[0].text == "Let me show you this is the kit"


def test_merge_close_segments_keeps_large_gaps_separate():
    segments = [
        RawSegment(start=0, end=5, text="What is 2 plus 2?"),
        RawSegment(start=10, end=11, text="Four"),  # 5s gap, real turn-taking
    ]
    merged = merge_close_segments(segments, max_gap=2.0)
    assert len(merged) == 2


def test_label_speakers_merges_before_labeling():
    # Three short chunks of one continuous teacher monologue, each < 6s
    # alone, but together they exceed the teacher threshold once merged.
    segments = [
        RawSegment(start=0, end=3, text="First we look inside."),
        RawSegment(start=3.3, end=6, text="It is a different pattern."),
        RawSegment(start=6.2, end=9, text="You will notice the shape."),
    ]
    labeled = label_speakers(segments)
    assert len(labeled) == 1
    assert labeled[0].speaker == "teacher"
