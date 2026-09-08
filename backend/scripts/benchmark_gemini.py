"""
Standalone Benchmark Script for Gemini Hybrid Pipeline vs Local Whisper Pipeline

Usage:
    cd backend
    python scripts/benchmark_gemini.py --audio path/to/recording.wav
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add app parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.pipeline import run_pipeline
from app.services.preprocessing import get_audio_duration


def run_benchmark(audio_path: str, output_report: str = "GEMINI_BENCHMARK_REPORT.md"):
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"RUNNING GEMINI HYBRID BENCHMARK ON: {audio_path}")
    print("=" * 60)

    clean_wav = f"storage/uploads/bench_clean_{int(time.time())}.wav"
    os.makedirs("storage/uploads", exist_ok=True)
    os.makedirs("storage/results", exist_ok=True)

    # Calculate Audio Duration
    from app.services.preprocessing import get_audio_duration, standardize_audio
    standardize_audio(audio_path, clean_wav)
    duration = get_audio_duration(clean_wav)
    print(f"Audio Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

    # 1. Run Gemini Hybrid Pipeline
    print("\n[1/2] Testing Gemini Hybrid Pipeline...")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    os.environ["ASR_PROVIDER"] = "gemini"
    
    t0 = time.time()
    gemini_result = run_pipeline(
        session_id=f"bench_gemini_{int(time.time())}",
        raw_path=audio_path,
        clean_path=clean_wav,
        update_status=lambda s, p: print(f"  Gemini Status: {s} | {p or ''}"),
    )
    gemini_time = time.time() - t0
    gemini_rtf = gemini_time / duration if duration > 0 else 0.0

    # 2. Run Local Faster-Whisper Pipeline
    print("\n[2/2] Testing Local Faster-Whisper Pipeline...")
    os.environ["ASR_PROVIDER"] = "whisper"
    
    t0 = time.time()
    whisper_result = run_pipeline(
        session_id=f"bench_whisper_{int(time.time())}",
        raw_path=audio_path,
        clean_path=clean_wav,
        update_status=lambda s, p: print(f"  Whisper Status: {s} | {p or ''}"),
    )
    whisper_time = time.time() - t0
    whisper_rtf = whisper_time / duration if duration > 0 else 0.0

    # Cleanup clean WAV
    try:
        os.remove(clean_wav)
    except Exception:
        pass

    # Generate Markdown Report
    gemini_segs = gemini_result.transcript.segments
    whisper_segs = whisper_result.transcript.segments

    report = f"""# Gemini Hybrid Pipeline Benchmark Report

## Recording Overview
- **Audio File**: `{audio_path}`
- **Duration**: {duration:.2f} seconds ({duration/60:.2f} minutes)
- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## ⚡ Performance & Real-Time Factor (RTF) Comparison

| Metric | Local Faster-Whisper | Gemini Hybrid Pipeline | Improvement |
|---|---|---|---|
| **Processing Time** | {whisper_time:.2f}s ({whisper_time/60:.2f}m) | **{gemini_time:.2f}s ({gemini_time/60:.2f}m)** | **{((whisper_time - gemini_time) / whisper_time * 100):.1f}% faster** |
| **Real-Time Factor (RTF)** | {whisper_rtf:.4f} | **{gemini_rtf:.4f}** | Lower is better |
| **Total Segments** | {len(whisper_segs)} | {len(gemini_segs)} | — |
| **Speaker Turns** | {whisper_result.metrics.speaker_turns} | {gemini_result.metrics.speaker_turns} | — |
| **Teacher Dominance** | {whisper_result.metrics.teacher_dominance_ratio*100:.1f}% | {gemini_result.metrics.teacher_dominance_ratio*100:.1f}% | — |

---

## 🎯 Quality & Accuracy Highlights

- **Stage 1 (Gemini ASR)**: Uses `gemini-2.5-flash` with zero forced English translation to accurately transcribe code-switched Hindi-English speech.
- **Stage 2 (Correction Agent)**: Analyzes sentence context and domain terms to correct phonetic misspellings without inventing speech.

### Sample Transcripts

#### Gemini Hybrid Pipeline (First 5 segments)
```text
"""
    for s in gemini_segs[:5]:
        report += f"[{s.start:.1f}s - {s.end:.1f}s] ({s.speaker}): {s.text}\n"

    report += """```

#### Local Faster-Whisper Pipeline (First 5 segments)
```text
"""
    for s in whisper_segs[:5]:
        report += f"[{s.start:.1f}s - {s.end:.1f}s] ({s.speaker}): {s.text}\n"

    report += "```\n"

    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETE! Report written to {output_report}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemini Hybrid vs Local Whisper Benchmark")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--output", default="GEMINI_BENCHMARK_REPORT.md", help="Output markdown report path")
    args = parser.parse_args()

    run_benchmark(args.audio, args.output)
