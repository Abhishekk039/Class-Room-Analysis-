"""
Standalone benchmark script for the classroom voice analytics pipeline.

Usage:
    cd backend
    python scripts/benchmark.py --audio path/to/recording.wav [--output TRANSCRIPTION_BENCHMARK.md]

Measures: total processing time, RTF, peak memory, chunk count,
segment count, suspicious segment rate, domain term hit rate.
Writes a markdown report without requiring the FastAPI server to be running.
"""
import argparse
import os
import sys
import time
import tracemalloc
import uuid
from pathlib import Path

# Ensure the backend package is importable when run from backend/.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import RESULTS_DIR, UPLOAD_DIR
from app.services.pipeline import run_pipeline


def benchmark(audio_path: str) -> dict:
    session_id = f"bench_{uuid.uuid4().hex[:8]}"
    raw_path   = str(Path(UPLOAD_DIR) / f"{session_id}_raw.wav")
    clean_path = str(Path(UPLOAD_DIR) / f"{session_id}_clean.wav")

    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    # Copy audio to upload dir so pipeline can find it.
    import shutil
    shutil.copy2(audio_path, raw_path)

    statuses = []
    def update_status(status, progress):
        entry = f"{status}" + (f" [{progress}]" if progress else "")
        statuses.append(entry)
        print(f"  → {entry}", flush=True)

    tracemalloc.start()
    t0 = time.perf_counter()

    result = run_pipeline(
        session_id=session_id,
        raw_path=raw_path,
        clean_path=clean_path,
        update_status=update_status,
    )

    elapsed = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    segs = result.transcript.segments
    suspicious = sum(1 for s in segs if s.suspicious)
    corrected  = sum(1 for s in segs if s.correction_applied)
    avg_conf   = (
        sum(s.confidence for s in segs if s.confidence is not None)
        / max(1, sum(1 for s in segs if s.confidence is not None))
    )

    return {
        "audio_path":      audio_path,
        "audio_duration":  result.transcript.duration_seconds,
        "processing_time": elapsed,
        "rtf":             elapsed / max(1, result.transcript.duration_seconds),
        "peak_memory_mb":  peak_mem / 1024 / 1024,
        "chunk_count":     result.chunk_count,
        "segment_count":   len(segs),
        "suspicious_rate": suspicious / max(1, len(segs)),
        "corrected_count": corrected,
        "avg_confidence":  avg_conf,
        "teacher_talk_pct": (
            result.metrics.teacher_talk_time_seconds
            / max(1, result.metrics.teacher_talk_time_seconds + result.metrics.student_talk_time_seconds)
            * 100
        ),
        "question_count":  result.metrics.teacher_question_count,
        "response_count":  result.metrics.student_response_count,
    }


def write_report(m: dict, output_path: str) -> None:
    audio_min = m["audio_duration"] / 60
    proc_min  = m["processing_time"] / 60

    report = f"""# TRANSCRIPTION_BENCHMARK.md

## Recording
- File: `{m["audio_path"]}`
- Duration: {audio_min:.1f} min ({m["audio_duration"]:.0f} s)

## Performance (v2 Optimized Pipeline)

| Metric | Value |
|---|---|
| Processing time | {proc_min:.1f} min ({m["processing_time"]:.1f} s) |
| Real-Time Factor (RTF) | {m["rtf"]:.4f} |
| Peak memory | {m["peak_memory_mb"]:.0f} MB |
| Chunks processed | {m["chunk_count"]} |

> RTF = processing_time / audio_duration.
> Lower is faster. Target: < 0.20 (process 67 min in < 13 min).

## Transcript Quality

| Metric | Value |
|---|---|
| Segments produced | {m["segment_count"]} |
| Average confidence | {m["avg_confidence"]:.3f} |
| Suspicious segments | {m["suspicious_rate"]*100:.1f}% |
| Domain corrections applied | {m["corrected_count"]} |

## Classroom Analytics

| Metric | Value |
|---|---|
| Teacher talk share | {m["teacher_talk_pct"]:.1f}% |
| Questions detected | {m["question_count"]} |
| Student responses | {m["response_count"]} |

## Before / After Comparison

| Metric | Before (v1) | After (v2) |
|---|---|---|
| Processing time | ~30 min | {proc_min:.1f} min |
| RTF | ~0.45 | {m["rtf"]:.4f} |
| Segments produced | 15 | {m["segment_count"]} |
| Whisper task | translate | transcribe |
| Beam size | 5 | 2 |
| Parallel chunks | No | Yes |
| Chunk caching | No | Yes |
| Domain vocabulary | No | Yes |
| Confidence scores | No | Yes |
| Repetition filter | No | Yes |

> WER cannot be computed without a human-verified ground-truth transcript.
> Label this section as an approximate/reference comparison only.

## Pipeline Status Log

```
{chr(10).join(m.get("statuses", []))}
```
"""
    Path(output_path).write_text(report, encoding="utf-8")
    print(f"\nReport written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the classroom pipeline.")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--output", default="TRANSCRIPTION_BENCHMARK.md")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"ERROR: audio file not found: {args.audio}")
        sys.exit(1)

    print(f"Benchmarking: {args.audio}")
    print("Running pipeline...")
    metrics = benchmark(args.audio)
    write_report(metrics, args.output)
    print(f"\nRTF={metrics['rtf']:.4f}  time={metrics['processing_time']:.1f}s  segments={metrics['segment_count']}")


if __name__ == "__main__":
    main()
