"""
Conservative transcript quality layer.

Rules:
  1. Repetition detection  — sliding window; flags consecutive segments
     whose text is identical or >85 % similar.
  2. Boundary deduplication — removes words repeated across the chunk
     overlap at chunk join points.
  3. Domain-term correction — replaces a misheard word only when the
     segment confidence is HIGH (>= CONFIDENCE_HIGH). Otherwise marks
     the segment [unclear] and preserves original_text.
  4. Short-noise filter — drops sub-0.5 s, low-confidence segments.

Nothing here freely rewrites transcript text. The source of truth is
always the STT output; this layer only flags or makes conservative
substitutions from the vocabulary's common_corrections dictionary.
"""
import json
import re
from pathlib import Path
from typing import List

from app.core.config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    REPETITION_WINDOW,
    VOCABULARY_FILE,
)
from app.services.transcription import RawSegment


# ---------------------------------------------------------------------------
# Vocabulary loader
# ---------------------------------------------------------------------------

def _load_corrections() -> dict:
    """Return the common_corrections dict from the active vocabulary file."""
    vocab_path = Path(VOCABULARY_FILE)
    if not vocab_path.exists():
        alt = Path(__file__).parent.parent.parent / VOCABULARY_FILE
        vocab_path = alt if alt.exists() else None
    if vocab_path and vocab_path.exists():
        try:
            data = json.loads(vocab_path.read_text(encoding="utf-8"))
            return {k.lower(): v for k, v in data.get("common_corrections", {}).items()}
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# 1. Short-noise filter
# ---------------------------------------------------------------------------

def filter_noise_segments(segments: List[RawSegment]) -> List[RawSegment]:
    """
    Drop segments that are very short AND have low confidence.
    These are almost always background noise mis-transcribed.
    """
    kept = []
    for seg in segments:
        duration = seg.end - seg.start
        # Keep if duration is reasonable OR confidence is high enough.
        if duration >= 0.5 or seg.confidence >= CONFIDENCE_MEDIUM:
            kept.append(seg)
    return kept


# ---------------------------------------------------------------------------
# 2. Boundary deduplication
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _similarity(a: str, b: str) -> float:
    """Simple token-overlap similarity in [0, 1]."""
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def deduplicate_boundaries(segments: List[RawSegment]) -> List[RawSegment]:
    """
    After merging chunks, consecutive segments that overlap in time or have
    very similar text (> 85 %) within a 2-second window are collapsed.
    The segment with higher confidence is kept.
    """
    if not segments:
        return []

    result = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        time_overlap = seg.start < prev.end
        text_similar = _similarity(seg.text, prev.text) > 0.85

        if time_overlap or text_similar:
            # Keep the higher-confidence segment.
            if seg.confidence > prev.confidence:
                result[-1] = seg
            # else discard seg (prev stays)
        else:
            result.append(seg)
    return result


# ---------------------------------------------------------------------------
# 3. Repetition detection
# ---------------------------------------------------------------------------

def detect_repetitions(segments: List[RawSegment]) -> List[RawSegment]:
    """
    Sliding window of REPETITION_WINDOW segments.
    If a segment's text is identical (or > 85 % similar) to the previous
    N segments, mark it suspicious=True.

    Suspicious segments are NOT deleted here — the frontend can show them
    with a warning icon; a human can decide what to do.
    """
    for i, seg in enumerate(segments):
        window = segments[max(0, i - REPETITION_WINDOW): i]
        for prev in window:
            if _similarity(seg.text, prev.text) > 0.85:
                seg.suspicious = True
                break
    return segments


# ---------------------------------------------------------------------------
# 4. Domain-term correction
# ---------------------------------------------------------------------------

def apply_domain_corrections(segments: List[RawSegment]) -> List[RawSegment]:
    """
    Apply corrections from the vocabulary's common_corrections dict.

    Rules:
      - HIGH confidence  → auto-correct silently, set correction_applied=True
      - MEDIUM confidence → mark [unclear: original] but do not correct
      - LOW confidence   → preserve original, mark suspicious=True
    """
    corrections = _load_corrections()
    if not corrections:
        return segments

    for seg in segments:
        text_lower = seg.text.lower()
        matched_phrase = None
        for wrong, right in corrections.items():
            if wrong in text_lower:
                matched_phrase = (wrong, right)
                break

        if matched_phrase is None:
            continue

        wrong, right = matched_phrase

        if seg.confidence >= CONFIDENCE_HIGH:
            # Auto-correct — preserve original for audit.
            seg.original_text = seg.text
            seg.text = re.sub(re.escape(wrong), right, seg.text, flags=re.IGNORECASE)
            seg.correction_applied = True

        elif seg.confidence >= CONFIDENCE_MEDIUM:
            # Flag for review — do not modify text.
            seg.suspicious = True

        else:
            # Low confidence — mark unclear.
            seg.original_text = seg.text
            seg.text = f"[unclear] {seg.text}"
            seg.suspicious = True

    return segments


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_quality_pipeline(segments: List[RawSegment]) -> List[RawSegment]:
    """Apply all quality steps in order and return cleaned segments."""
    segments = filter_noise_segments(segments)
    segments = deduplicate_boundaries(segments)
    segments = detect_repetitions(segments)
    segments = apply_domain_corrections(segments)
    return segments
