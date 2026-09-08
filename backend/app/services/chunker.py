"""
VAD-aware audio chunker.

Splits a preprocessed (16 kHz mono WAV) audio file into short overlapping
slices that respect silence boundaries. Each slice is saved as a temporary
WAV file and described by a ChunkSpec that carries the original timestamps.

Design rules:
  - Never split mid-word: breaks happen only at detected silence regions.
  - Target chunk length: CHUNK_TARGET_SECONDS (~25 s).
  - Overlap: CHUNK_OVERLAP_SECONDS (~1.5 s) on each side so words at
    boundaries are captured by both adjacent chunks. The quality layer
    removes the resulting duplicates after transcription.
  - Falls back to fixed-interval splitting if VAD returns no segments.
  - Each ChunkSpec records the original start/end so timestamps in the
    final transcript are always relative to the original recording.
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from app.core.config import (
    CHUNK_OVERLAP_SECONDS,
    CHUNK_TARGET_SECONDS,
)


@dataclass
class ChunkSpec:
    chunk_id: int
    start: float      # seconds in the ORIGINAL recording
    end: float        # seconds in the ORIGINAL recording
    path: str         # absolute path to the sliced WAV file


def _merge_vad_regions(
    regions: List[Tuple[float, float]],
    target: float,
) -> List[Tuple[float, float]]:
    """
    Group consecutive VAD speech regions into chunks whose total duration
    is as close to `target` seconds as possible without splitting a region.

    Returns a list of (chunk_start, chunk_end) tuples in ascending order.
    """
    if not regions:
        return []

    chunks: List[Tuple[float, float]] = []
    current_start = regions[0][0]
    current_end   = regions[0][1]

    for start, end in regions[1:]:
        projected_duration = end - current_start
        if projected_duration <= target:
            # Extend current chunk to include this region (and the gap).
            current_end = end
        else:
            chunks.append((current_start, current_end))
            current_start = start
            current_end   = end

    chunks.append((current_start, current_end))
    return chunks


def _fallback_fixed_chunks(
    total_duration: float,
    target: float,
) -> List[Tuple[float, float]]:
    """
    When VAD provides no speech regions, split the audio into fixed
    `target`-second intervals.
    """
    chunks = []
    start = 0.0
    while start < total_duration:
        end = min(start + target, total_duration)
        chunks.append((start, end))
        start = end
    return chunks


def _extract_slice(
    source_path: str,
    start: float,
    end: float,
    out_path: str,
) -> None:
    """
    Extract [start, end] seconds from source_path into out_path using
    ffmpeg stream copy (no re-encoding — very fast).
    """
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t",  str(duration),
        "-i",  source_path,
        "-c",  "copy",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg slice [{start:.2f}–{end:.2f}] failed: {result.stderr}"
        )


def build_chunks(
    audio_path: str,
    vad_regions: List[Tuple[float, float]],
    total_duration: float,
    chunk_dir: str,
    session_id: str,
) -> List[ChunkSpec]:
    """
    Build ChunkSpec objects for a preprocessed audio file.

    Parameters
    ----------
    audio_path     : path to the preprocessed 16 kHz mono WAV
    vad_regions    : (start, end) speech regions from preprocessing.get_vad_segments()
    total_duration : total audio duration in seconds
    chunk_dir      : directory where slice WAV files will be written
    session_id     : used to name slice files uniquely

    Returns
    -------
    List of ChunkSpec sorted by chunk_id (== chronological order).
    """
    Path(chunk_dir).mkdir(parents=True, exist_ok=True)

    if vad_regions:
        raw_chunks = _merge_vad_regions(vad_regions, CHUNK_TARGET_SECONDS)
    else:
        raw_chunks = _fallback_fixed_chunks(total_duration, CHUNK_TARGET_SECONDS)

    specs: List[ChunkSpec] = []
    for idx, (start, end) in enumerate(raw_chunks):
        # Apply overlap: extend each chunk boundary outward by CHUNK_OVERLAP_SECONDS
        # but clamp to [0, total_duration].
        slice_start = max(0.0, start - CHUNK_OVERLAP_SECONDS)
        slice_end   = min(total_duration, end + CHUNK_OVERLAP_SECONDS)

        out_path = str(Path(chunk_dir) / f"{session_id}_chunk_{idx:04d}.wav")
        _extract_slice(audio_path, slice_start, slice_end, out_path)

        specs.append(ChunkSpec(
            chunk_id=idx,
            start=start,
            end=end,
            path=out_path,
        ))

    return specs
