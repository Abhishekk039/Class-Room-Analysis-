"""
Classroom Voice Analytics Pipeline

Deployment configuration:
    - Gemini is the only ASR provider.
    - Local Whisper fallback is disabled.
    - This prevents Render's 512 MB instance from loading
      the memory-heavy local Whisper transcription pipeline.

Pipeline:
    1. Preprocess audio using FFmpeg
    2. Transcribe audio using Gemini
    3. Run Stage 2 contextual correction
    4. Run speaker diarization
    5. Detect questions
    6. Calculate engagement metrics
    7. Generate summary
    8. Persist final analysis result
"""

import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.core.config import (
    ASR_PROVIDER,
    GEMINI_API_KEY,
    RESULTS_DIR,
    WHISPER_LANGUAGE,
)

from app.models.schemas import (
    AnalysisResult,
    EngagementMetrics,
    Transcript,
    TranscriptSegment,
)

from app.services.analysis import (
    compute_silence,
    compute_talk_time,
    count_student_responses,
    detect_questions,
)

from app.services.diarization import label_speakers

from app.services.metrics import (
    interaction_density,
    student_participation_indicator,
    teacher_dominance_ratio,
)

from app.services.preprocessing import (
    get_audio_duration,
    standardize_audio,
)

from app.services.summary import generate_summary


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_result(result: AnalysisResult) -> None:
    """
    Save the completed analysis result to disk.
    """

    path = Path(RESULTS_DIR) / f"{result.session_id}.json"

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )


def load_result(
    session_id: str,
) -> Optional[AnalysisResult]:
    """
    Load a previously saved analysis result.
    """

    path = Path(RESULTS_DIR) / f"{session_id}.json"

    if not path.exists():
        return None

    try:
        return AnalysisResult.model_validate_json(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        logger.exception(
            "Failed to load saved result for session %s",
            session_id,
        )

        return None


def load_all_results() -> Dict[str, AnalysisResult]:
    """
    Load all previously saved analysis results.

    This is used when the backend starts again.
    """

    results: Dict[str, AnalysisResult] = {}

    results_dir = Path(RESULTS_DIR)

    if not results_dir.exists():
        return results

    for path in results_dir.glob("*.json"):

        session_id = path.stem

        result = load_result(
            session_id
        )

        if result is not None:
            results[session_id] = result

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    session_id: str,
    raw_path: str,
    clean_path: str,
    update_status: Callable[
        [str, Optional[str]],
        None,
    ],
) -> AnalysisResult:
    """
    Run the complete Classroom Voice Analytics pipeline.

    The deployed version intentionally uses Gemini as the only
    transcription provider.

    Parameters
    ----------
    session_id:
        Unique ID for the analysis session.

    raw_path:
        Path to the uploaded audio file.

    clean_path:
        Path where the cleaned WAV file will be created.

    update_status:
        Callback used to update frontend job status.

    Returns
    -------
    AnalysisResult
        Completed analysis result.
    """

    t_start = time.time()

    # -----------------------------------------------------------------------
    # 1. Audio preprocessing
    # -----------------------------------------------------------------------

    update_status(
        "preprocessing",
        None,
    )

    logger.info(
        "Starting audio preprocessing for session %s",
        session_id,
    )

    standardize_audio(
        raw_path,
        clean_path,
    )

    duration = get_audio_duration(
        clean_path
    )

    logger.info(
        "Audio preprocessing completed for session %s. Duration: %.2f seconds",
        session_id,
        duration,
    )

    # -----------------------------------------------------------------------
    # 2. Gemini configuration
    # -----------------------------------------------------------------------

    provider_setting = os.getenv(
        "ASR_PROVIDER",
        ASR_PROVIDER or "gemini",
    ).lower().strip()

    gemini_key = os.getenv(
        "GEMINI_API_KEY",
        GEMINI_API_KEY,
    )

    # Gemini is required for the deployed backend.
    if provider_setting not in (
        "gemini",
        "hybrid",
    ):
        raise RuntimeError(
            "This deployment requires ASR_PROVIDER=gemini."
        )

    if not gemini_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured on the backend."
        )

    # -----------------------------------------------------------------------
    # 3. Gemini transcription
    # -----------------------------------------------------------------------

    update_status(
        "transcribing_gemini",
        "Stage 1: Gemini ASR in progress...",
    )

    logger.info(
        "Starting Gemini transcription for session %s",
        session_id,
    )

    all_segments = []

    try:

        from app.services.transcription_service import (
            transcribe_audio_gemini,
        )

        all_segments = transcribe_audio_gemini(
            clean_path,
            api_key=gemini_key,
        )

    except Exception as e:

        logger.exception(
            "Gemini Stage 1 ASR failed for session %s",
            session_id,
        )

        # IMPORTANT:
        # Do NOT fall back to local Whisper.
        #
        # Local Whisper can exceed Render's 512 MB memory limit.
        raise RuntimeError(
            f"Gemini transcription failed: {e}"
        ) from e

    # -----------------------------------------------------------------------
    # 4. Validate Gemini result
    # -----------------------------------------------------------------------

    if not all_segments:

        raise RuntimeError(
            "Gemini returned no speech segments. "
            "Please check that the recording contains audible speech "
            "and try again."
        )

    logger.info(
        "Gemini transcription completed for session %s. "
        "Segments: %d",
        session_id,
        len(all_segments),
    )

    used_provider = "gemini"

    # -----------------------------------------------------------------------
    # 5. Stage 2 contextual correction
    # -----------------------------------------------------------------------

    update_status(
        "stage2_correction",
        "Stage 2: Contextual correction agent running...",
    )

    logger.info(
        "Starting Stage 2 correction for session %s",
        session_id,
    )

    try:

        from app.services.corrector import (
            run_stage2_correction,
        )

        all_segments = run_stage2_correction(
            all_segments
        )

    except Exception as e:

        logger.exception(
            "Stage 2 correction failed for session %s",
            session_id,
        )

        raise RuntimeError(
            f"Stage 2 correction failed: {e}"
        ) from e

    # -----------------------------------------------------------------------
    # 6. Speaker diarization
    # -----------------------------------------------------------------------

    update_status(
        "diarization",
        None,
    )

    logger.info(
        "Starting speaker diarization for session %s",
        session_id,
    )

    try:

        labeled: List[
            TranscriptSegment
        ] = label_speakers(
            all_segments
        )

    except Exception as e:

        logger.exception(
            "Diarization failed for session %s",
            session_id,
        )

        raise RuntimeError(
            f"Speaker diarization failed: {e}"
        ) from e

    # -----------------------------------------------------------------------
    # 7. Question detection
    # -----------------------------------------------------------------------

    labeled = detect_questions(
        labeled
    )

    # -----------------------------------------------------------------------
    # 8. Metrics
    # -----------------------------------------------------------------------

    update_status(
        "computing_metrics",
        None,
    )

    logger.info(
        "Computing engagement metrics for session %s",
        session_id,
    )

    teacher_time, student_time = compute_talk_time(
        labeled
    )

    silence = compute_silence(
        labeled,
        duration,
    )

    question_count = sum(
        1
        for segment in labeled
        if segment.is_question
    )

    response_count = count_student_responses(
        labeled
    )

    student_turns = sum(
        1
        for segment in labeled
        if segment.speaker == "student"
    )

    total_turns = len(labeled)

    # -----------------------------------------------------------------------
    # 9. Engagement metrics
    # -----------------------------------------------------------------------

    metrics = EngagementMetrics(

        teacher_talk_time_seconds=teacher_time,

        student_talk_time_seconds=student_time,

        silence_seconds=silence,

        teacher_question_count=question_count,

        student_response_count=response_count,

        teacher_dominance_ratio=teacher_dominance_ratio(
            teacher_time,
            teacher_time + student_time,
        ),

        student_participation_indicator=student_participation_indicator(
            student_turns,
            total_turns,
        ),

        interaction_density_per_minute=interaction_density(
            question_count,
            response_count,
            duration,
        ),

        speaker_turns=total_turns,
    )

    # -----------------------------------------------------------------------
    # 10. Processing statistics
    # -----------------------------------------------------------------------

    t_end = time.time()

    processing_time = round(
        t_end - t_start,
        2,
    )

    rtf = (
        round(
            processing_time / duration,
            4,
        )
        if duration > 0
        else 0.0
    )

    # -----------------------------------------------------------------------
    # 11. Transcript
    # -----------------------------------------------------------------------

    transcript = Transcript(

        session_id=session_id,

        duration_seconds=duration,

        language=WHISPER_LANGUAGE or "auto",

        segments=labeled,
    )

    # -----------------------------------------------------------------------
    # 12. Final analysis result
    # -----------------------------------------------------------------------

    result = AnalysisResult(

        session_id=session_id,

        transcript=transcript,

        metrics=metrics,

        summary=generate_summary(
            metrics
        ),

        processing_time_seconds=processing_time,

        rtf=rtf,

        # Gemini processes the complete audio,
        # so there is no local Whisper chunk count.
        chunk_count=1,

        asr_provider=used_provider,

        stage2_applied=any(
            segment.correction_applied
            for segment in all_segments
        ),
    )

    # -----------------------------------------------------------------------
    # 13. Save result
    # -----------------------------------------------------------------------

    save_result(
        result
    )

    logger.info(
        "Pipeline completed successfully for session %s "
        "in %.2f seconds",
        session_id,
        processing_time,
    )

    return result