# 🎙️ Classroom Voice Analytics

project-link : https://class-room-analysis.vercel.app/



## 📸 Project Screenshots


![Classroom Analysis](./Screenshot%202026-09-09%20111008.png)


![Classroom Analysis](./Screenshot%202026-09-09%20111109.png)


![Classroom Analysis](./Screenshot%202026-09-09%20113050.png)


![Classroom Analysis](./Screenshot%202026-09-09%20113135.png)

> Offline-first classroom audio analysis system for transcription, interaction detection, speaker labeling, and engagement insights.

This project is a working MVP for analyzing classroom recordings and converting them into structured educational insights. The backend accepts uploaded audio, preprocesses it, transcribes speech, extracts classroom interactions, and calculates engagement metrics.

The current system is designed for multilingual classroom speech, especially English, Hindi, and Hindi-English code-switching, while keeping the main processing local using faster-whisper and FFmpeg.

---

## 📌 Current Status

The project is currently in an MVP / prototype stage, but it already implements a real end-to-end pipeline:

- Audio upload through FastAPI
- FFmpeg-based preprocessing
- VAD-based chunk generation
- Parallel chunk transcription
- Quality filtering and correction passes
- Speaker-label heuristic
- Question and response detection
- Engagement metric calculation
- Classroom summary generation
- result caching and resumable processing

This is a strong technical foundation, but it is not yet a production-grade classroom analytics platform.

---

## 🚀 What the System Does Today

### 1. Audio preprocessing
The backend standardizes uploaded recordings before transcription.

Current implementation:

- converts to mono
- resamples to 16 kHz
- normalizes volume
- prepares clean audio for speech detection and ASR

### 2. Speech transcription
The transcription stage uses local whisper-based processing.

Current implementation:

- uses faster-whisper
- supports local CPU transcription
- supports configurable model size and compute settings
- optionally falls back to Gemini-based ASR when configured
- runs chunked transcription for long recordings

### 3. VAD and chunking
The system detects speech regions and splits long files into smaller segments for better handling.

Current implementation:

- VAD regions are used to build speech chunks
- target chunk length is around 25 seconds
- overlapping windows are added to avoid missing words at chunk boundaries
- chunks are processed in parallel

### 4. Parallel processing
The pipeline is designed for long classroom recordings and does not transcribe the entire file as one single block.

Current implementation:

- stores temporary chunk files
- uses ThreadPoolExecutor for chunk-level transcription
- merges the results in chronological order
- saves chunk-level cache data so interrupted processing can resume

### 5. Correction and quality filtering
After transcription, the project runs a quality pipeline to reduce common transcription issues.

Current implementation:

- removes repeated segments
- catches suspicious or noisy outputs
- applies correction logic based on context and domain vocabulary
- improves transcript cleanliness before analytics

### 6. Classroom analysis
Once clean transcript text is available, the project computes classroom-related signals.

Current implementation:

- teacher vs student labeling using a duration heuristic
- question detection
- student response detection
- talk time analysis
- silence analysis
- engagement metrics
- summary generation

### 7. API and persistence
The API exposes the system to clients and maintains job state.

Current implementation:

- FastAPI backend
- upload endpoint
- status endpoint
- analysis retrieval endpoint
- results saved to storage/results
- chunk cache for resumable processing

---

## ⏱️ Audio Processing Time: Current Reality

At the moment, long classroom audio processing can take around 17 minutes on CPU-based setups, depending on:

- audio length
- CPU power
- chosen Whisper model size
- number of chunks
- whether Gemini fallback is enabled
- quality and correction stage cost
- amount of speech activity in the recording

This is not unusual for a local CPU transcription pipeline with multiple stages. In other words, the project is currently accurate and functional, but it is slower than a production real-time system.

### Why it takes time

The main factors are:

- Whisper inference on CPU is relatively slow for long recordings
- segmentation and chunk overlap create additional audio slicing
- each chunk is processed individually
- post-processing adds correction and analysis steps
- full transcript quality checks are not free

---

## ⚡ What We Should Do to Make It Faster and Better

The path forward is clear: the system should become faster without losing transcript quality.

### Immediate performance improvements

1. Optimize Whisper model choice
   - use smaller or quantized model for faster inference when quality can remain acceptable
   - tune model size based on classroom complexity

2. Improve chunking efficiency
   - reduce unnecessary overlap
   - tune chunk length and VAD parameters for faster but still accurate transcription
   - avoid excessive slicing for short or low-activity recordings

3. Increase parallelism intelligently
   - increase worker count for multi-core CPUs
   - batch stronger processing where possible
   - avoid oversubscription on low-resource machines

4. Use GPU support where available
   - if CUDA or GPU runtime is available, ASR speed improves substantially
   - this is one of the biggest speed gains for the current setup

5. Avoid redundant processing
   - keep cache and resume logic stronger
   - skip reprocessing unchanged segments and failed jobs more cleanly

### Accuracy and quality improvements

1. Replace heuristic diarization
   - use pyannote.audio or speaker embedding-based diarization
   - classify teacher/student more reliably than duration-only logic

2. Improve multilingual handling
   - improve Hindi and code-switch recognition quality
   - add domain-specific vocabulary tuning for classroom subjects

3. Strengthen correction logic
   - use better confidence scoring
   - improve context-based correction passes
   - integrate more subject-aware language rules

4. Add robust evaluation metrics
   - compare transcription quality against reference transcripts
   - track repeated phrase and hallucination rates

---

## 🧭 Future Scope and Roadmap

The project has a clear path from MVP to production-ready classroom intelligence platform.

### Phase 1 — Speed and Stability

- optimize chunking overhead
- tune Whisper configuration for CPU efficiency
- improve cache and resume behavior
- reduce processing time from ~17 minutes toward a more practical target
- support more efficient background job execution

### Phase 2 — Better Speech Understanding

- replace heuristic speaker labels with true speaker diarization
- integrate speaker embeddings and clustering
- improve question detection quality
- improve multilingual code-switch recognition
- add better confidence and uncertainty scoring

### Phase 3 — Classroom Intelligence

- track teacher dominance, student participation, and interaction density across sessions
- compare classroom sessions over time
- provide historical trends for teacher engagement patterns
- add better summary generation for classroom reports

### Phase 4 — Product Platform

- add database-backed persistence for sessions, transcripts, and metrics
- add authentication and user roles
- add analytics dashboard for teachers and administrators
- add historical session comparisons and teacher/student reports
- support batch processing for multiple recordings

### Phase 5 — Real-time and Large-Scale Deployment

- add near real-time or streaming audio analysis
- support live classroom monitoring
- deploy backend to cloud or edge infrastructure
- enable scaling across multiple classrooms and sessions
- integrate with LMS or classroom management systems

---

## 🏗️ Current Architecture

```text
Classroom audio
      ↓
Upload API
      ↓
Preprocessing (FFmpeg)
      ↓
VAD + Chunking
      ↓
Parallel Transcription
      ↓
Quality / Correction Layer
      ↓
Diarization Heuristic
      ↓
Question / Response Analysis
      ↓
Engagement Metrics
      ↓
Summary Generation
      ↓
Storage / API Output
```

---

## 🧩 Current Implementation Details

This project already includes the following components:

- FastAPI backend for audio upload and status tracking
- FFmpeg preprocessing pipeline
- VAD-based segmentation logic
- 25-second chunk generation with overlap
- ThreadPoolExecutor parallel chunk processing
- local faster-whisper transcription
- optional Gemini ASR as a fallback or hybrid path
- stage-2 correction pipeline
- quality checks for repetitive or suspicious outputs
- simple teacher/student heuristic labeling
- metrics for participation and interaction density
- summary generation for classroom sessions
- caching of chunk outputs to disk

---

## 📊 Current Detection Logic

### Teacher vs student labeling
The current model uses a simple heuristic:

- longer speech segments are treated as teacher speech
- shorter segments are treated as student speech

This is a practical MVP approximation, but it is not a true diarization system.

### Question detection
Questions are inferred using a combination of:

- question mark punctuation
- question keywords such as what, why, when, who, how, can you, and similar patterns

### Response detection
A student response is detected when a student segment occurs within a short time window after a teacher question.

---

## ⚠️ Current Limitations

The current version is intentionally an MVP and has clear constraints.

### 1. Speaker diarization is approximate
The system is not using true speaker separation yet. It depends on duration-based assumptions.

### 2. Multilingual accuracy can vary
Hindi, Hinglish, and noisy classroom speech can still cause errors in punctuation, phrasing, and turn labeling.

### 3. Processing can be slow on CPU
The total time for some long recordings is around 17 minutes, which makes it less suitable for real-time or classroom-scale live processing today.

### 4. Persistence is still basic
Results are stored in local result folders, but a production database layer is still a future step.

### 5. Some correction logic is heuristic
It works well for common classroom patterns, but it is not a full language-quality system.

---

## 🔌 API Overview

### Upload audio
```http
POST /api/upload
```

Uploads a classroom recording and starts processing.

### Check status
```http
GET /api/status/{session_id}
```

### Retrieve results
```http
GET /api/analysis/{session_id}
```

Supported audio input formats usually include:

- WAV
- MP3
- M4A
- MP4
- AAC
- FLAC
- OGG
- WebM

---

## 🧪 Testing

The project contains automated tests focused on core logic.

Run:

```bash
cd backend
pytest -v
```

The system is designed so that important business logic can be validated independently of raw audio files or full model execution.

---

## ⚙️ Setup

### Prerequisites

- Python 3.10+
- FFmpeg
- Git

### Install

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
# source venv/bin/activate
pip install -r requirements.txt
```

### Run API

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

### Docker

```bash
docker compose up --build
```

---

## 🔐 Privacy and Design Philosophy

The project is designed with offline-first processing in mind. Audio is processed locally when possible, which keeps classroom recordings away from unnecessary third-party transmission.

This makes the architecture suitable for privacy-sensitive educational environments, even though some setup and model-download steps still require initial access to packages or model files.

---

## ✅ Final Position

This project is already a strong MVP with a real end-to-end pipeline, but it still sits between a research prototype and a production classroom intelligence product.

The biggest current issue is processing time: long recordings can take around 17 minutes on CPU, mostly because of local Whisper transcription and chunk-based processing. The best path forward is to combine model optimization, GPU support, better diarization, and stronger multilingual accuracy while keeping the system privacy-conscious and locally deployable.

This is a good foundation for a future classroom analytics platform, not just a one-off proof of concept.

---

## 👨‍💻 Author

Abhishek

Built with Python, FastAPI, faster-whisper, FFmpeg, and practical classroom AI workflows.
