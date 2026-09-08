import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import UPLOAD_DIR
from app.services.preprocessing import standardize_audio, get_audio_duration
from app.services.transcription import transcribe_audio
from app.services.diarization import label_speakers
from app.services.analysis import (
    detect_questions,
    count_student_responses,
    compute_talk_time,
    compute_silence,
)
from app.services.metrics import (
    teacher_dominance_ratio,
    student_participation_indicator,
    interaction_density,
)
from app.services.summary import generate_summary
from app.models.schemas import Transcript, EngagementMetrics, AnalysisResult, UploadResponse

router = APIRouter()

# In-memory store for this MVP - swap for a real DB/file store for production use.
_results: dict[str, AnalysisResult] = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    raw_path = f"{UPLOAD_DIR}/{session_id}_raw{Path(file.filename).suffix}"
    clean_path = f"{UPLOAD_DIR}/{session_id}_clean.wav"

    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    try:
        standardize_audio(raw_path, clean_path)
        duration = get_audio_duration(clean_path)

        raw_segments, language = transcribe_audio(clean_path)
        labeled_segments = label_speakers(raw_segments)
        labeled_segments = detect_questions(labeled_segments)

        teacher_time, student_time = compute_talk_time(labeled_segments)
        silence = compute_silence(labeled_segments, duration)
        question_count = sum(1 for s in labeled_segments if s.is_question)
        response_count = count_student_responses(labeled_segments)

        student_turns = sum(1 for s in labeled_segments if s.speaker == "student")
        total_turns = len(labeled_segments)

        metrics = EngagementMetrics(
            teacher_talk_time_seconds=teacher_time,
            student_talk_time_seconds=student_time,
            silence_seconds=silence,
            teacher_question_count=question_count,
            student_response_count=response_count,
            teacher_dominance_ratio=teacher_dominance_ratio(teacher_time, teacher_time + student_time),
            student_participation_indicator=student_participation_indicator(student_turns, total_turns),
            interaction_density_per_minute=interaction_density(question_count, response_count, duration),
        )

        transcript = Transcript(
            session_id=session_id,
            duration_seconds=duration,
            language=language,
            segments=labeled_segments,
        )

        result = AnalysisResult(
            session_id=session_id,
            transcript=transcript,
            metrics=metrics,
            summary=generate_summary(metrics),
        )
        _results[session_id] = result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    return UploadResponse(session_id=session_id, message="Processed successfully")


@router.get("/analysis/{session_id}", response_model=AnalysisResult)
async def get_analysis(session_id: str):
    result = _results.get(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    return result
