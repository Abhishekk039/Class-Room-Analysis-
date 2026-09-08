"""
Unified Audio Transcription Service (Stage 1 ASR)

Supports:
  1. Gemini API ASR (gemini-2.5-flash) via official google-genai SDK
  2. Local Faster-Whisper ASR (fallback / offline mode)
"""
import json
import logging
import os
import time
from typing import List, Optional

from app.core.config import (
    ASR_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    VOCABULARY_FILE,
)
from app.services.transcription import RawSegment, transcribe_audio as transcribe_audio_whisper

logger = logging.getLogger(__name__)


def _load_vocabulary_terms(custom_vocabulary: Optional[List[str]] = None) -> List[str]:
    """Helper to merge custom vocabulary with standard domain vocabulary JSON."""
    terms = []
    if custom_vocabulary:
        terms.extend(custom_vocabulary)
    
    if os.path.exists(VOCABULARY_FILE):
        try:
            with open(VOCABULARY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                terms.extend(data.get("terms", []))
        except Exception as e:
            logger.warning(f"Could not load domain vocabulary from {VOCABULARY_FILE}: {e}")
            
    return list(dict.fromkeys(terms))  # Remove duplicates preserving order


def transcribe_audio_gemini(
    audio_path: str,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    custom_vocabulary: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> List[RawSegment]:
    """
    Transcribes audio using Gemini File API and gemini-2.5-flash.
    Returns structured timestamped segments with speaker detection and confidence.
    """
    key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set.")

    # Modern google-genai SDK import with fallback
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        sdk_type = "genai"
    except ImportError:
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=key)
            sdk_type = "legacy"
        except ImportError:
            raise ImportError("Neither 'google-genai' nor 'google-generativeai' SDK is installed.")

    terms = _load_vocabulary_terms(custom_vocabulary)
    vocab_str = ", ".join(terms) if terms else "Standard classroom terminology"

    prompt = f"""
    You are an expert audio transcription system specializing in classroom lectures.
    
    INSTRUCTIONS:
    1. Transcribe the spoken audio with precise start and end timestamps in seconds.
    2. Maintain exact spoken speech, including Indian English accents, numbers, technical terms, and code-switched phrases (Hindi/English). Do NOT force translate non-English speech to English.
    3. Identify speaker roles: "teacher" (explanatory, instruction) or "student" (answers, short questions).
    4. Flag questions accurately (is_question: true/false).
    5. Technical domain vocabulary context: [{vocab_str}]. Pay special attention to correct spelling of these technical terms and numbers.
    6. Return ONLY a JSON object in the following format:

    {{
      "segments": [
        {{
          "start": 0.0,
          "end": 4.5,
          "text": "Good morning class, today we will cover Kirchhoff's circuit law.",
          "speaker": "teacher",
          "is_question": false,
          "confidence": 0.95
        }}
      ]
    }}
    """

    uploaded_file = None
    try:
        logger.info(f"Uploading audio file {audio_path} to Gemini File API...")
        
        if sdk_type == "genai":
            uploaded_file = client.files.upload(file=audio_path)
            
            # Wait for file processing if needed
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)
                
            if uploaded_file.state.name == "FAILED":
                raise RuntimeError(f"Gemini File API processing failed: {uploaded_file.error.message}")

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            response_text = response.text
        else:
            # Legacy SDK
            uploaded_file = legacy_genai.upload_file(audio_path)
            model = legacy_genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content([uploaded_file, prompt])
            response_text = response.text

        # Parse output
        data = json.loads(response_text)
        segments_json = data.get("segments", [])
        
        raw_segments = []
        for s in segments_json:
            raw_segments.append(
                RawSegment(
                    start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)),
                    text=str(s.get("text", "")).strip(),
                    confidence=float(s.get("confidence", 0.90)),
                )
            )
        
        logger.info(f"Gemini transcription successful: {len(raw_segments)} segments generated.")
        return raw_segments

    finally:
        # Cleanup uploaded file from Gemini File API
        if uploaded_file:
            try:
                if sdk_type == "genai":
                    client.files.delete(name=uploaded_file.name)
                else:
                    legacy_genai.delete_file(uploaded_file.name)
                logger.info("Cleaned up temporary Gemini uploaded file.")
            except Exception as cleanup_err:
                logger.warning(f"File cleanup warning: {cleanup_err}")


def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    custom_vocabulary: Optional[List[str]] = None,
    provider: Optional[str] = None,
) -> List[RawSegment]:
    """
    Unified transcription entrypoint.
    Decides between Gemini API and local Faster-Whisper.
    """
    target_provider = (provider or ASR_PROVIDER or "gemini").lower()
    gemini_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")

    if target_provider in ("gemini", "hybrid") and gemini_key:
        try:
            logger.info("Starting Stage 1 ASR: Gemini API...")
            return transcribe_audio_gemini(
                audio_path=audio_path,
                language=language,
                domain=domain,
                custom_vocabulary=custom_vocabulary,
                api_key=gemini_key,
            )
        except Exception as e:
            logger.error(f"Gemini transcription failed ({e}). Falling back to local Whisper pipeline...")

    logger.info("Using local Faster-Whisper ASR pipeline...")
    return transcribe_audio_whisper(audio_path)
