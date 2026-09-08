from typing import List, Optional
from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None        # "teacher" | "student"
    is_question: bool = False
    # v2 additions — all optional so existing tests / API consumers still work
    confidence: Optional[float] = None   # [0, 1] from avg_logprob
    suspicious: bool = False             # repetition / low-confidence flag
    correction_applied: bool = False     # domain-term correction was made
    original_text: Optional[str] = None  # text before correction


class Transcript(BaseModel):
    session_id: str
    duration_seconds: float
    language: Optional[str] = None
    segments: List[TranscriptSegment]


class EngagementMetrics(BaseModel):
    teacher_talk_time_seconds: float
    student_talk_time_seconds: float
    silence_seconds: float
    teacher_question_count: int
    student_response_count: int
    teacher_dominance_ratio: float
    student_participation_indicator: float
    interaction_density_per_minute: float
    # v2 addition
    speaker_turns: int = 0


class AnalysisResult(BaseModel):
    session_id: str
    transcript: Transcript
    metrics: EngagementMetrics
    summary: str
    # v2 & Gemini hybrid additions
    processing_time_seconds: float = 0.0
    rtf: float = 0.0                     # processing_time / audio_duration
    chunk_count: int = 0
    asr_provider: Optional[str] = "gemini"
    stage2_applied: bool = False



class UploadResponse(BaseModel):
    session_id: str
    message: str
    status: str = "queued"


class JobStatus(BaseModel):
    session_id: str
    status: str
    progress: Optional[str] = None       # e.g. "42/180 chunks"
    error: Optional[str] = None
