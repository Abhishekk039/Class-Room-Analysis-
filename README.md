# Classroom Voice Analytics MVP

An offline-first prototype that turns classroom audio into a transcript,
speaker-labeled turns, and engagement metrics — built for MakerGhat's
Full Stack Developer pre-work assignment (Task 1).

## Project structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI app entrypoint
│   ├── api/routes.py            # POST /api/upload, GET /api/analysis/{id}
│   ├── services/
│   │   ├── preprocessing.py     # ffmpeg: resample to 16kHz mono + loudness normalize
│   │   ├── transcription.py     # faster-whisper wrapper
│   │   ├── diarization.py       # teacher/student heuristic
│   │   ├── analysis.py          # question detection, response counting, talk time, silence
│   │   ├── metrics.py           # the 3 engagement metric formulas
│   │   └── summary.py           # rule-based classroom summary
│   ├── models/schemas.py        # Pydantic request/response models
│   └── core/config.py           # thresholds, model settings, keyword lists
├── tests/                       # pytest — 13 tests, all pure-logic (no audio needed to run)
├── Dockerfile
└── requirements.txt
docker-compose.yml
```

## Development approach

Built bottom-up, de-risking the least familiar part first:
1. Transcription (faster-whisper) proven on sample audio before anything else
2. Speaker-turn heuristic and question/response detection as pure, independently
   testable functions
3. Engagement metrics as one function per metric, each unit tested
4. API wiring last, once the underlying logic was verified correct

## Engagement metrics

| Metric | Formula / logic | Explanation | Interpretation |
|---|---|---|---|
| Teacher Dominance Ratio | `teacher_talk_time / total_talk_time` | Share of all speech that came from the teacher | >0.7 lecture-heavy, <0.4 discussion-heavy |
| Student Participation Indicator | `student_turns / total_turns` | Share of speaking turns taken by students | Higher = participation spread across students, not just the teacher |
| Interaction Density | `(teacher_questions + student_responses) / duration_minutes` | Rate of question-and-response exchanges per minute | Higher = a more active, back-and-forth session |

## Assumptions made

- **Speaker diarization is a documented heuristic, not real diarization.**
  Segments longer than 6 seconds are labeled "teacher" (continuous
  explanatory speech), shorter segments "student" (brief answers). This
  is an explicit approximation — the brief allows this ("approximation is
  acceptable"). A production version would use `pyannote.audio` speaker
  embeddings instead; noted here rather than hidden.
- **Question detection** combines two signals: the segment ends with "?"
  (Whisper often restores punctuation) OR starts with a question keyword
  ("what", "why", "can you", etc.). Combining both catches cases where
  punctuation restoration misses.
- **Student response** = any student segment starting within 8 seconds of
  a detected teacher question ending.
- **Offline-first**: faster-whisper runs fully locally once model weights
  are cached (one-time download on first run, no network needed after).
  Chosen over the brief's listed Google Speech-to-Text option specifically
  because that's a cloud API and would violate the offline-first
  requirement stated in the assignment background.
- **Language**: auto-detect is enabled by default; can be pinned via
  `WHISPER_LANGUAGE` in config for a known-language dataset.

## Limitations / what I'd do with more time

- Swap the duration-based heuristic for real speaker diarization
  (`pyannote.audio`) with speaker-embedding clustering
- Add voice activity detection (VAD) as an explicit preprocessing step to
  also expose silence-duration as a first-class segment type, not just a
  derived total
- Persist results to a real datastore instead of an in-memory dict
- Add a confidence score per transcript segment surfaced in the UI
- Noisy code-switched recordings can still produce repeated or semantically
  incorrect phrases even after filtering and language pinning. For example,
  one real run repeated "The copper is gone." three times. This is a
  multilingual Whisper model-quality limitation, not something the speaker
  duration heuristic can reliably correct; a larger model or better recording
  is required for further transcription accuracy.

## Running locally

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Or with Docker:

```bash
docker compose up --build
```

Then `POST` an audio file to `http://localhost:8000/api/upload`. Supported
formats are WAV, MP3, M4A, MP4, AAC, FLAC, OGG, and WebM up to 2 GiB. Uploads
are streamed to disk and processed in the background, so the request returns a
session ID immediately. Poll `/api/status/{session_id}` until it is
`completed`, then fetch `/api/analysis/{session_id}`.

The transcription pipeline uses faster-whisper's multilingual model with
automatic language detection and `translate` mode. Hindi speech, English
speech, and Hindi-English code-switching are converted to English transcript
text. Audio preprocessing converts recordings to mono 16 kHz, removes low and
high-frequency noise, applies noise reduction, and normalizes loudness. For
CPU-only machines, `small` provides a quality-oriented default; use `tiny` or
`base` in `backend/app/core/config.py` when turnaround time matters more than
accuracy.

## Running tests

```bash
cd backend
pytest -v
```

All 13 tests are pure-logic (metrics + analysis functions) and run without
needing faster-whisper or any audio file.
