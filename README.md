# 🎙️ Classroom Voice Analytics

> **Offline-first AI system for classroom transcription, speaker analysis, and engagement insights.**

Classroom Voice Analytics is an AI-powered backend system that converts classroom audio into **English transcripts, speaker-labeled conversation turns, and quantitative engagement metrics**.

The system is designed for environments where audio may contain **English, Hindi, or Hindi-English code-switching**, while keeping the core transcription pipeline local through `faster-whisper`.

---

## 🚀 Key Features

* 🎙️ **Automatic Speech Transcription** using `faster-whisper`
* 🌐 **Multilingual Audio Support** with automatic language detection
* 🔄 **Hindi → English Translation** for multilingual classroom recordings
* 👨‍🏫 **Teacher / Student Speaker Classification**
* ❓ **Automatic Question Detection**
* 💬 **Student Response Detection**
* 📊 **Classroom Engagement Metrics**
* 🔇 **Silence Analysis**
* 📝 **Automated Classroom Summary**
* ⚡ **Background Audio Processing**
* 💾 **Offline-first processing**
* 🧪 **13 automated unit tests**
* 🐳 **Docker support**
* 🚀 **FastAPI REST API**

---

# 🧠 System Overview

The system processes classroom recordings through a multi-stage pipeline:

```text
                    Classroom Audio
                          │
                          ▼
                ┌───────────────────┐
                │ Audio Upload API  │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Audio Preprocessing│
                │                   │
                │ • Mono conversion │
                │ • 16 kHz resample  │
                │ • Noise reduction │
                │ • Loudness normalize│
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Speech-to-Text    │
                │  faster-whisper   │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Speaker Analysis  │
                │                   │
                │ Teacher / Student │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Conversation      │
                │ Analysis          │
                │                   │
                │ • Questions       │
                │ • Responses       │
                │ • Silence         │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Engagement Metrics│
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Classroom Summary │
                └───────────────────┘
```

---

# 📊 Engagement Analytics

The MVP calculates three primary engagement indicators.

| Metric                              | Formula                                                      | Interpretation                                           |
| ----------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| **Teacher Dominance Ratio**         | `teacher_talk_time / total_talk_time`                        | Measures how much of the session is teacher-led          |
| **Student Participation Indicator** | `student_turns / total_turns`                                | Measures the proportion of speaking turns from students  |
| **Interaction Density**             | `(teacher_questions + student_responses) / duration_minutes` | Measures the frequency of question-response interactions |

### Teacher Dominance Ratio

```text
> 0.70  → Lecture-heavy
0.40–0.70 → Mixed interaction
< 0.40  → Discussion-heavy
```

### Student Participation

A higher value indicates that a greater proportion of speaking turns are attributed to students.

### Interaction Density

A higher value indicates more frequent back-and-forth interaction between the teacher and students.

---

# 🧩 Detection Pipeline

## 1. Audio Preprocessing

Incoming recordings are processed using `ffmpeg`.

The preprocessing stage:

* Converts audio to mono
* Resamples to **16 kHz**
* Applies noise reduction
* Removes low/high-frequency noise
* Normalizes loudness

This provides a more consistent input for transcription.

---

## 2. Speech Transcription

The project uses **faster-whisper** for local speech recognition.

The pipeline supports:

```text
English
Hindi
Hindi-English Code Switching
```

The system can automatically detect the spoken language and translate the resulting speech into English transcript text.

For CPU-only systems:

```text
small → Quality-oriented default
base  → Faster processing
tiny  → Fastest / lowest resource usage
```

The model can be configured through the project configuration.

---

## 3. Speaker Classification

The MVP uses a documented heuristic to distinguish between teacher and student speech.

```text
Segment duration > 6 seconds
        ↓
     Teacher

Segment duration ≤ 6 seconds
        ↓
     Student
```

### Why this approach?

Continuous explanatory speech is more likely to belong to the teacher, while shorter segments often represent student responses.

This is intentionally treated as an **approximation rather than true speaker diarization**.

### Production Improvement

A production implementation would replace this heuristic with speaker embeddings and clustering using technologies such as:

```text
pyannote.audio
        +
Speaker Embeddings
        +
Clustering
```

---

# ❓ Question & Response Detection

Question detection combines multiple signals.

### Signal 1 — Punctuation

The transcript segment ends with:

```text
?
```

### Signal 2 — Question Keywords

The segment begins with words such as:

```text
what
why
when
where
who
how
can you
could you
do you
```

Combining both approaches improves detection when punctuation restoration is imperfect.

---

## 💬 Student Response Detection

A student segment is considered a response when it begins within **8 seconds** of a detected teacher question ending.

```text
Teacher Question
       │
       │ ≤ 8 seconds
       ▼
Student Speech
       │
       ▼
Student Response
```

---

# 🏗️ Project Architecture

```text
Classroom-Voice-Analytics/
│
├── backend/
│   │
│   ├── app/
│   │   ├── main.py
│   │   │
│   │   ├── api/
│   │   │   └── routes.py
│   │   │
│   │   ├── services/
│   │   │   ├── preprocessing.py
│   │   │   ├── transcription.py
│   │   │   ├── diarization.py
│   │   │   ├── analysis.py
│   │   │   ├── metrics.py
│   │   │   └── summary.py
│   │   │
│   │   ├── models/
│   │   │   └── schemas.py
│   │   │
│   │   └── core/
│   │       └── config.py
│   │
│   ├── tests/
│   │
│   ├── Dockerfile
│   └── requirements.txt
│
├── docker-compose.yml
└── README.md
```

---

# 🔌 API

The backend exposes a REST API through **FastAPI**.

## Upload Audio

```http
POST /api/upload
```

Uploads a classroom audio recording and starts background processing.

### Supported formats

```text
WAV
MP3
M4A
MP4
AAC
FLAC
OGG
WebM
```

Maximum upload size:

```text
2 GiB
```

The upload endpoint immediately returns a `session_id`.

---

## Check Processing Status

```http
GET /api/status/{session_id}
```

Example:

```json
{
  "session_id": "bd691f39-414a-4049-ae19-58e380d109b5",
  "status": "completed"
}
```

Possible states include:

```text
processing
completed
failed
```

---

## Retrieve Analysis

```http
GET /api/analysis/{session_id}
```

Once processing is complete, the analysis endpoint returns the generated transcript, speaker turns, detected interactions, metrics, and classroom summary.

---

# 🛠️ Technology Stack

| Technology         | Purpose                     |
| ------------------ | --------------------------- |
| **Python**         | Core backend and processing |
| **FastAPI**        | REST API                    |
| **faster-whisper** | Speech recognition          |
| **FFmpeg**         | Audio preprocessing         |
| **Pydantic**       | API schemas and validation  |
| **Pytest**         | Automated testing           |
| **Docker**         | Containerized deployment    |

---

# 💻 Development Approach

The project was developed **bottom-up**, starting with the highest-risk component and progressively integrating the system.

### Phase 1 — Transcription

Validated `faster-whisper` against sample classroom recordings.

### Phase 2 — Analysis Logic

Implemented speaker classification and question/response detection as independently testable functions.

### Phase 3 — Engagement Metrics

Implemented each metric as an isolated function with unit tests.

### Phase 4 — API Integration

Connected the validated processing pipeline to FastAPI endpoints.

This approach reduced integration risk and made the core analysis logic easier to test independently.

---

# 🧪 Testing

The project currently contains:

```text
13 automated tests
```

The tests focus on pure logic and do not require audio files or model downloads.

Run:

```bash
cd backend
pytest -v
```

Example:

```text
13 passed
```

---

# ⚙️ Installation

## Prerequisites

Make sure the following are installed:

* Python 3.10+
* FFmpeg
* Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/classroom-voice-analytics.git
cd classroom-voice-analytics
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Start the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🐳 Docker

The application can also be started using Docker Compose:

```bash
docker compose up --build
```

---

# 🔄 Typical Workflow

```text
1. Upload classroom recording
             ↓
2. Receive session ID
             ↓
3. Audio preprocessing
             ↓
4. Speech transcription
             ↓
5. Speaker classification
             ↓
6. Question detection
             ↓
7. Response detection
             ↓
8. Engagement calculation
             ↓
9. Classroom summary
             ↓
10. Retrieve analysis through API
```

---

# ⚠️ Current Limitations

This is an **MVP**, so several components are intentionally simplified.

### Speaker Diarization

The current system uses a duration-based heuristic instead of true speaker diarization.

### Voice Activity Detection

Silence is currently derived from transcript timing rather than being represented as a dedicated VAD segment.

### Data Persistence

Analysis results are currently maintained in memory rather than a persistent database.

### Transcription Quality

Noisy or heavily code-switched recordings can produce:

* Repeated phrases
* Incorrect words
* Semantically inaccurate segments

For example, a recording may occasionally repeat a phrase such as:

```text
"The copper is gone."
```

multiple times.

This is primarily a speech-recognition/model-quality limitation and cannot be reliably solved through speaker-duration heuristics alone.

---

# 🔮 Future Roadmap

* [ ] Replace heuristic speaker detection with `pyannote.audio`
* [ ] Add explicit Voice Activity Detection
* [ ] Add persistent database storage
* [ ] Add transcript confidence scores
* [ ] Add speaker embeddings
* [ ] Improve multilingual/code-switching accuracy
* [ ] Add real-time audio processing
* [ ] Add classroom analytics dashboard
* [ ] Add historical session comparison
* [ ] Add teacher/student participation trends
* [ ] Add authentication and role-based access
* [ ] Deploy production backend

---

# 🎯 Use Cases

Classroom Voice Analytics can be extended for:

* 🏫 Classroom engagement monitoring
* 👨‍🏫 Teacher-led session analysis
* 📚 Educational research
* 📊 Student participation analysis
* 🎙️ Automated lecture transcription
* 🔎 Classroom interaction analysis
* 📈 Long-term engagement tracking

---

# 🔐 Privacy & Offline-First Design

A key design goal of this MVP is **local audio processing**.

Once the required Whisper model weights are available locally, transcription can run without sending classroom recordings to a third-party speech API.

This makes the architecture suitable for environments where minimizing external audio-data transmission is important.

> Note: the initial model download requires network access; subsequent inference can run locally with the cached model.

---

# 📌 Assignment Context

This project was developed as part of the **MakerGhat Full Stack Developer pre-work assignment — Task 1**.

The implementation prioritizes:

* Functional correctness
* Clear engineering decisions
* Testable business logic
* Offline-first processing
* Documented assumptions
* Extensible architecture

---

# 👨‍💻 Author

**Abhishek**

Built with Python, FastAPI, faster-whisper, FFmpeg, and a focus on practical AI-assisted classroom analytics.

---

## ⭐ Project Status

**Status:** MVP / Prototype

The current implementation demonstrates the complete pipeline from:

```text
Audio → Transcription → Speaker Analysis → Engagement Metrics → Summary
```

while clearly documenting the areas that would require more sophisticated approaches for production deployment.
