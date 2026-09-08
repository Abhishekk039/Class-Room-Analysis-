"""
Unit tests for Gemini Hybrid Pipeline (Stage 1 ASR & Stage 2 Contextual Corrector)
"""
import os
from unittest.mock import MagicMock, patch

import pytest
from app.services.corrector import run_stage2_correction
from app.services.transcription import RawSegment
from app.services.transcription_service import transcribe_audio, transcribe_audio_gemini


def test_transcribe_audio_falls_back_to_whisper_when_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    with patch("app.services.transcription_service.transcribe_audio_whisper") as mock_whisper:
        mock_whisper.return_value = [
            RawSegment(start=0.0, end=3.0, text="Hello class", confidence=0.90)
        ]
        
        result = transcribe_audio("fake_path.wav", provider="gemini")
        assert len(result) == 1
        assert result[0].text == "Hello class"
        mock_whisper.assert_called_once()


def test_stage2_correction_passes_through_if_no_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    input_segments = [
        RawSegment(start=0.0, end=2.0, text="Today we will discuss home law.", confidence=0.85)
    ]
    
    result = run_stage2_correction(input_segments)
    assert len(result) == 1
    assert result[0].text == "Today we will discuss home law."
