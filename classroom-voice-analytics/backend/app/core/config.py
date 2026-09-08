"""
Central configuration for the classroom analytics backend.
Kept as plain constants (not pydantic-settings) to minimize dependencies
for this MVP; swap for BaseSettings later if env-based config grows.
"""

# --- Transcription ---
WHISPER_MODEL_SIZE = "small"          # "tiny" | "base" | "small" | "medium"
WHISPER_DEVICE = "cpu"                # "cpu" | "cuda"
WHISPER_COMPUTE_TYPE = "int8"         # int8 is fastest on CPU
WHISPER_LANGUAGE = "hi"               # pin explicitly instead of auto-detect for code-switched Hindi/English audio; set None to auto-detect
WHISPER_BEAM_SIZE = 1                 # 1 = greedy/fast (dev), 5 = higher quality/slower (final run)
WHISPER_CPU_THREADS = 0               # 0 = let faster-whisper pick; set to os.cpu_count() to force full core usage

# --- Diarization heuristic ---
# Segments longer than this many seconds are more likely to be the teacher
# (lecture-style explanation). This is an explicit, documented approximation,
# not real speaker-embedding diarization. See docs/ARCHITECTURE.md.
TEACHER_SEGMENT_DURATION_THRESHOLD_SECONDS = 6.0

# Gap (in seconds) below which two consecutive Whisper segments are merged
# into one turn, on the assumption it's a mid-monologue breath pause rather
# than a real speaker change. Tune this against your actual sample audio -
# real turn-taking gaps (someone else responds) tend to run longer than
# natural breathing pauses within one person's speech.
MERGE_GAP_THRESHOLD_SECONDS = 2.0

# --- Question / response detection ---
QUESTION_KEYWORDS = (
    "what", "why", "how", "when", "where", "who", "which",
    "can you", "could you", "do you", "does anyone", "who can",
)

# A student segment counts as a "response" if it starts within this many
# seconds of a teacher question segment ending.
RESPONSE_WINDOW_SECONDS = 8.0

# --- Audio preprocessing ---
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
HIGHPASS_FREQUENCY_HZ = 100           # cuts low-frequency rumble (fans, handling noise, traffic) below speech range
SILENCE_TRIM_THRESHOLD_DB = "-35dB"   # trims long dead-air stretches at start/end, not mid-speech pauses
SILENCE_TRIM_MIN_DURATION = "1"       # seconds of silence before ffmpeg trims it

# --- Storage ---
UPLOAD_DIR = "storage/uploads"
RESULTS_DIR = "storage/results"
