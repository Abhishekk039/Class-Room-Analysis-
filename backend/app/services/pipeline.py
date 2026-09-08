"""
Parallel transcription pipeline orchestrator.

Replaces the flat sequential block that was previously embedded in
api/routes._process_audio(). Responsibilities:

  1. Preprocess audio (ffmpeg → 16 kHz mono WAV)
  2. Run VAD to find speech regions
  3. Build VAD-aligned chunks (~25 s each)
  4. Transcribe chunks in parallel using ThreadPoolExecutor
  5. Merge chunk results, sorted by chunk_id (= chronological order)
  6. Run the quality pipeline (dedup, repetition, domain corrections)
  7. Run diarization heuristic
  8. Run analysis (questions, talk time, silence, metrics, summary)
  9. Persist result to disk; update job status with progress

Chunk-level caching: each completed chunk is written to
  storage/results/{session_id}/chunk_{id:04d}.json
If the process crashes and resumes, finished chunks are loaded from cache
rather than re-transcribed, enabling resumable processing.
"""
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


from app.core.config import (
    ASR_PROVIDER,
    GEMINI_API_KEY,
    MAX_WORKERS,
    RESULTS_DIR,
    UPLOAD_DIR,
    WHISPER_LANGUAGE,
)

from app.models.schemas import (
    AnalysisResult,
    EngagementMetrics,
    JobStatus,
    Transcript,
    TranscriptSegment,
)
from app.services.analysis import (
    compute_silence,
    compute_talk_time,
    count_student_responses,
    detect_questions,
)
from app.services.chunker import ChunkSpec, build_chunks
from app.services.diarization import label_speakers
from app.services.metrics import (
    interaction_density,
    student_participation_indicator,
    teacher_dominance_ratio,
)
from app.services.preprocessing import (
    get_audio_duration,
    get_vad_segments,
    standardize_audio,
)
from app.services.quality import run_quality_pipeline
from app.services.summary import generate_summary
from app.services.transcription import RawSegment, transcribe_chunk


# ---------------------------------------------------------------------------
# Chunk-level cache helpers
# ---------------------------------------------------------------------------

def _chunk_cache_path(session_id: str, chunk_id: int) -> Path:
    return Path(RESULTS_DIR) / session_id / f"chunk_{chunk_id:04d}.json"


def _save_chunk_cache(session_id: str, chunk_id: int, segments: List[RawSegment]) -> None:
    path = _chunk_cache_path(session_id, chunk_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "start": s.start, "end": s.end, "text": s.text,
            "confidence": s.confidence, "suspicious": s.suspicious,
            "correction_applied": s.correction_applied,
            "original_text": s.original_text,
        }
        for s in segments
    ]
    path.write_text(json.dumps(data), encoding="utf-8")


def _load_chunk_cache(session_id: str, chunk_id: int) -> Optional[List[RawSegment]]:
    path = _chunk_cache_path(session_id, chunk_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [RawSegment(**d) for d in data]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_result(result: AnalysisResult) -> None:
    path = Path(RESULTS_DIR) / f"{result.session_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def load_result(session_id: str) -> Optional[AnalysisResult]:
    path = Path(RESULTS_DIR) / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return AnalysisResult.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_all_results() -> Dict[str, AnalysisResult]:
    """Called at startup to reload persisted results into memory."""
    results: Dict[str, AnalysisResult] = {}
    results_dir = Path(RESULTS_DIR)
    if not results_dir.exists():
        return results
    for p in results_dir.glob("*.json"):
        sid = p.stem
        r = load_result(sid)
        if r:
            results[sid] = r
    return results


# ---------------------------------------------------------------------------
# Single-chunk worker (runs in thread pool)
# ---------------------------------------------------------------------------

def _process_chunk(
    chunk: ChunkSpec,
    session_id: str,
) -> tuple[int, List[RawSegment]]:
    """
    Transcribe one chunk. Returns (chunk_id, segments).
    Loads from cache if already processed (resume support).
    """
    cached = _load_chunk_cache(session_id, chunk.chunk_id)
    if cached is not None:
        return chunk.chunk_id, cached

    segments = transcribe_chunk(chunk)
    _save_chunk_cache(session_id, chunk.chunk_id, segments)

    # Clean up the temporary slice file.
    try:
        Path(chunk.path).unlink(missing_ok=True)
    except OSError:
        pass

    return chunk.chunk_id, segments


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    session_id: str,
    raw_path: str,
    clean_path: str,
    update_status: Callable[[str, Optional[str]], None],
) -> AnalysisResult:
    """
    Full end-to-end pipeline for one recording session.

    Parameters
    ----------
    session_id    : unique session identifier
    raw_path      : path to the uploaded raw audio file
    clean_path    : path where the preprocessed WAV will be written
    update_status : callback(status_str, progress_str) → updates _jobs in routes.py

    Returns
    -------
    AnalysisResult ready to store and return via the API.
    """
    t_start = time.time()
    chunk_dir = str(Path(UPLOAD_DIR) / session_id / "chunks")

    # 1 — Preprocess
    update_status("preprocessing", None)
    standardize_audio(raw_path, clean_path)
    duration = get_audio_duration(clean_path)

    # 2 — Speech & Transcription (Stage 1)
    provider_setting = os.getenv("ASR_PROVIDER", ASR_PROVIDER or "gemini").lower()
    gemini_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
    
    all_segments: List[RawSegment] = []
    total_chunks = 1
    used_provider = "whisper"

    if provider_setting in ("gemini", "hybrid") and gemini_key:
        try:
            update_status("transcribing_gemini", "Stage 1: Gemini ASR in progress...")
            from app.services.transcription_service import transcribe_audio_gemini
            all_segments = transcribe_audio_gemini(clean_path, api_key=gemini_key)
            used_provider = "gemini"
        except Exception as e:
            logger.warning(f"Gemini Stage 1 ASR failed ({e}); falling back to local Whisper...")
            all_segments = []

    if not all_segments:
        # Fallback / Local Whisper chunked pipeline
        update_status("analyzing_speech", None)
        vad_regions = get_vad_segments(clean_path)

        update_status("chunking", None)
        chunks = build_chunks(clean_path, vad_regions, duration, chunk_dir, session_id)
        total_chunks = len(chunks)

        update_status("transcribing", f"0/{total_chunks} chunks")
        completed = 0
        workers = min(MAX_WORKERS, os.cpu_count() or 2, total_chunks)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_chunk, chunk, session_id): chunk
                for chunk in chunks
            }
            chunk_results: Dict[int, List[RawSegment]] = {}
            for future in as_completed(futures):
                chunk_id, segs = future.result()
                chunk_results[chunk_id] = segs
                completed += 1
                update_status("transcribing", f"{completed}/{total_chunks} chunks")

        for chunk_id in sorted(chunk_results.keys()):
            all_segments.extend(chunk_results[chunk_id])
        all_segments.sort(key=lambda s: (s.start, s.end))

        if all_segments:
            used_provider = "whisper"

    if not all_segments:
        raise RuntimeError(
            "No speech was transcribed from the audio. "
            "Check that the recording contains audible speech and try again."
        )

    # 5 — Stage 2 Contextual Correction Agent / Quality pipeline
    update_status("stage2_correction", "Stage 2: Contextual correction agent running...")
    from app.services.corrector import run_stage2_correction
    all_segments = run_stage2_correction(all_segments)


    # 6 — Diarization
    update_status("diarization", None)
    labeled: List[TranscriptSegment] = label_speakers(all_segments)
    labeled = detect_questions(labeled)

    # 7 — Metrics
    update_status("computing_metrics", None)
    teacher_time, student_time = compute_talk_time(labeled)
    silence = compute_silence(labeled, duration)
    question_count = sum(1 for s in labeled if s.is_question)
    response_count = count_student_responses(labeled)
    student_turns = sum(1 for s in labeled if s.speaker == "student")
    total_turns = len(labeled)

    metrics = EngagementMetrics(
        teacher_talk_time_seconds=teacher_time,
        student_talk_time_seconds=student_time,
        silence_seconds=silence,
        teacher_question_count=question_count,
        student_response_count=response_count,
        teacher_dominance_ratio=teacher_dominance_ratio(
            teacher_time, teacher_time + student_time
        ),
        student_participation_indicator=student_participation_indicator(
            student_turns, total_turns
        ),
        interaction_density_per_minute=interaction_density(
            question_count, response_count, duration
        ),
        speaker_turns=total_turns,
    )

    t_end = time.time()
    processing_time = round(t_end - t_start, 2)
    rtf = round(processing_time / duration, 4) if duration > 0 else 0.0

    transcript = Transcript(
        session_id=session_id,
        duration_seconds=duration,
        language=WHISPER_LANGUAGE or "auto",
        segments=labeled,
    )

    result = AnalysisResult(
        session_id=session_id,
        transcript=transcript,
        metrics=metrics,
        summary=generate_summary(metrics),
        processing_time_seconds=processing_time,
        rtf=rtf,
        chunk_count=total_chunks,
        asr_provider=used_provider,
        stage2_applied=any(segment.correction_applied for segment in all_segments),
    )

    save_result(result)
    return result
