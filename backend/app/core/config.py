"""
Central configuration for the classroom analytics backend.
Kept as plain constants (not pydantic-settings) to minimize dependencies
for this MVP; swap for BaseSettings later if env-based config grows.

OPTIMIZATION NOTES (v2):
  - WHISPER_TASK changed translate → transcribe: eliminates the primary
    source of hallucinations and 20-30 % of CPU time.
  - WHISPER_BEAM_SIZE reduced 5 → 2: 30-40 % speed gain on CPU with
    negligible quality loss for classroom speech.
  - WHISPER_CONDITION_ON_PREVIOUS_TEXT = False: prevents a hallucination
    in one chunk from cascading into the next.
  - WHISPER_COMPRESSION_RATIO_THRESHOLD lowered to 1.8: forces Whisper to
    retry or drop highly repetitive outputs (fixes "No Can you hear me?"
    cascade seen in real output).
  - Chunking constants added so the 67-minute file is split into ~25-second
    VAD-aligned slices and processed in parallel.
"""
import os
from pathlib import Path

# Load .env file automatically
env_paths = [
    Path(__file__).parent.parent.parent / ".env",
    Path(__file__).parent.parent.parent.parent / ".env",
]
for ep in env_paths:
    if ep.exists():
        for line in ep.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE    = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE        = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE  = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # fastest on CPU

# ---------------------------------------------------------------------------
# Gemini API & Hybrid Pipeline Settings
# ---------------------------------------------------------------------------
GEMINI_API_KEY           = os.getenv("GEMINI_API_KEY", "")
ASR_PROVIDER             = os.getenv("ASR_PROVIDER", "whisper")  # "gemini" | "whisper" | "hybrid"
GEMINI_MODEL             = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ENABLE_STAGE2_CORRECTION = os.getenv("ENABLE_STAGE2_CORRECTION", "true").lower() == "true"


# "transcribe" keeps original Hindi/English text → far fewer hallucinations.
# Previously "translate" caused Whisper to fabricate English translations of
# silence and noise, producing the 22-minute "We are going back" segments.
_whisper_language = os.getenv("WHISPER_LANGUAGE", "auto").strip().lower()
WHISPER_LANGUAGE      = None if _whisper_language in {"", "auto", "detect"} else _whisper_language
WHISPER_TASK          = os.getenv("WHISPER_TASK", "transcribe")

# Beam size 2 is the best CPU trade-off: ~35 % faster than beam=5,
# quality difference is negligible for Hindi classroom speech.
WHISPER_BEAM_SIZE     = int(os.getenv("WHISPER_BEAM_SIZE", "2"))

# Probability below which a segment is considered non-speech and dropped.
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.5"))

# Repetition ratio above which Whisper falls back to temperature sampling.
# Default is 2.4; 1.8 catches the repeated-phrase hallucination pattern.
WHISPER_COMPRESSION_RATIO_THRESHOLD = float(
    os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "1.8")
)

# False = each chunk is decoded independently; prevents one bad chunk from
# poisoning subsequent chunks through the conditioning context.
WHISPER_CONDITION_ON_PREVIOUS_TEXT = (
    os.getenv("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false").lower() == "true"
)

WHISPER_VAD_MIN_SILENCE_MS = int(os.getenv("WHISPER_VAD_MIN_SILENCE_MS", "500"))

# ---------------------------------------------------------------------------
# Domain vocabulary
# ---------------------------------------------------------------------------
# Path to the active vocabulary JSON (relative to the backend/ directory).
# Swap for biology.json, cs.json, etc. to change subject.
VOCABULARY_FILE = os.getenv(
    "VOCABULARY_FILE", "app/vocabularies/electronics.json"
)

# ---------------------------------------------------------------------------
# Chunking & parallel processing
# ---------------------------------------------------------------------------
# Target duration (seconds) for each audio chunk fed to Whisper.
# VAD boundaries are respected; actual chunk length may vary ±5 s.
CHUNK_TARGET_SECONDS  = int(os.getenv("CHUNK_TARGET_SECONDS", "25"))

# Overlap added to each side of a chunk boundary so words at edges are not
# lost. The deduplicator in quality.py removes the resulting duplicates.
CHUNK_OVERLAP_SECONDS = float(os.getenv("CHUNK_OVERLAP_SECONDS", "1.5"))

# Number of worker threads for parallel chunk transcription.
# Defaults to min(4, cpu_count) at runtime; override here for tuning.
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))

# ---------------------------------------------------------------------------
# Diarization heuristic
# ---------------------------------------------------------------------------
# Segments longer than this many seconds are labeled "teacher".
# This is a documented approximation; see README for caveats.
TEACHER_SEGMENT_DURATION_THRESHOLD_SECONDS = float(
    os.getenv("TEACHER_SEGMENT_DURATION_THRESHOLD_SECONDS", "6.0")
)
MERGE_GAP_THRESHOLD_SECONDS = float(os.getenv("MERGE_GAP_THRESHOLD_SECONDS", "1.0"))
MAX_MERGE_DURATION_SECONDS  = float(os.getenv("MAX_MERGE_DURATION_SECONDS", "20.0"))

# ---------------------------------------------------------------------------
# Question / response detection
# ---------------------------------------------------------------------------
QUESTION_KEYWORDS = (
    "what", "why", "how", "when", "where", "who", "which",
    "can you", "could you", "do you", "does anyone", "who can",
)
RESPONSE_WINDOW_SECONDS = float(os.getenv("RESPONSE_WINDOW_SECONDS", "8.0"))

# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE      = 16000
TARGET_CHANNELS         = 1
AUDIO_MAX_UPLOAD_BYTES  = 2 * 1024 * 1024 * 1024   # 2 GiB
AUDIO_CHUNK_SIZE_BYTES  = 1024 * 1024

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
UPLOAD_DIR  = os.getenv("UPLOAD_DIR", "storage/uploads")
RESULTS_DIR = os.getenv("RESULTS_DIR", "storage/results")

# ---------------------------------------------------------------------------
# Quality / hallucination filter
# ---------------------------------------------------------------------------
# Consecutive segments with identical text within this window are flagged.
REPETITION_WINDOW = int(os.getenv("REPETITION_WINDOW", "3"))

# Confidence thresholds for automatic correction vs. review vs. preserve.
CONFIDENCE_HIGH   = float(os.getenv("CONFIDENCE_HIGH", "0.70"))
CONFIDENCE_MEDIUM = float(os.getenv("CONFIDENCE_MEDIUM", "0.50"))
