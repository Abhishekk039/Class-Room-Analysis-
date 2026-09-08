import os
import uuid
from pathlib import Path


from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException

from app.core.config import AUDIO_CHUNK_SIZE_BYTES, AUDIO_MAX_UPLOAD_BYTES, UPLOAD_DIR
from app.models.schemas import (
    AnalysisResult,
    JobStatus,
    UploadResponse,
)
from app.services.pipeline import load_all_results, run_pipeline, save_result

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory stores — results are also persisted to disk by pipeline.py
# so they survive server restarts (loaded in load_persisted_results below).
# ---------------------------------------------------------------------------
_results: dict[str, AnalysisResult] = {}
_jobs: dict[str, JobStatus] = {}
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".ogg", ".webm"}


def load_persisted_results() -> None:
    """Called once at startup to reload results that survived a restart."""
    loaded = load_all_results()
    _results.update(loaded)
    for sid in loaded:
        _jobs[sid] = JobStatus(session_id=sid, status="completed")


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _update_status(session_id: str, status: str, progress: str | None) -> None:
    _jobs[session_id] = JobStatus(
        session_id=session_id,
        status=status,
        progress=progress,
    )


def _process_audio(session_id: str, raw_path: str, clean_path: str) -> None:
    _update_status(session_id, "processing", None)
    try:
        result = run_pipeline(
            session_id=session_id,
            raw_path=raw_path,
            clean_path=clean_path,
            update_status=lambda status, progress: _update_status(
                session_id, status, progress
            ),
        )
        _results[session_id] = result
        _jobs[session_id] = JobStatus(
            session_id=session_id,
            status="completed",
            progress=f"{result.chunk_count}/{result.chunk_count} chunks",
        )
    except Exception as error:
        _jobs[session_id] = JobStatus(
            session_id=session_id, status="failed", error=str(error)
        )
    finally:
        for path in (raw_path, clean_path):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Endpoints — same signatures as before
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    extension = Path(file.filename or "").suffix.lower()
    if extension not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported audio format")

    raw_path   = f"{UPLOAD_DIR}/{session_id}_raw{extension}"
    clean_path = f"{UPLOAD_DIR}/{session_id}_clean.wav"

    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    with open(raw_path, "wb") as output:
        while chunk := await file.read(AUDIO_CHUNK_SIZE_BYTES):
            bytes_written += len(chunk)
            if bytes_written > AUDIO_MAX_UPLOAD_BYTES:
                Path(raw_path).unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Audio file exceeds the 2 GiB limit")
            output.write(chunk)

    _jobs[session_id] = JobStatus(session_id=session_id, status="queued")
    background_tasks.add_task(_process_audio, session_id, raw_path, clean_path)
    return UploadResponse(
        session_id=session_id,
        message="Audio queued for processing",
        status="queued",
    )


@router.get("/status/{session_id}", response_model=JobStatus)
async def get_status(session_id: str):
    status = _jobs.get(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    return status


@router.get("/analysis/{session_id}", response_model=AnalysisResult)
async def get_analysis(session_id: str):
    result = _results.get(session_id)
    if not result:
        job = _jobs.get(session_id)
        if job and job.status in {"queued", "processing", "preprocessing",
                                   "analyzing_speech", "chunking",
                                   "transcribing", "transcribing_gemini", "stage2_correction",
                                   "quality_check", "diarization", "computing_metrics"}:
            raise HTTPException(status_code=202, detail="Audio is still processing")
        if job and job.status == "failed":
            raise HTTPException(status_code=500, detail=job.error or "Audio processing failed")
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@router.get("/config")
async def get_config():
    """Returns active ASR provider and Gemini configuration status."""
    from app.core.config import ASR_PROVIDER, ENABLE_STAGE2_CORRECTION, GEMINI_MODEL
    key = os.getenv("GEMINI_API_KEY", "")
    return {
        "asr_provider": ASR_PROVIDER,
        "gemini_model": GEMINI_MODEL,
        "has_gemini_key": bool(key and key != "your_gemini_api_key_here"),
        "stage2_correction_enabled": ENABLE_STAGE2_CORRECTION,
    }


