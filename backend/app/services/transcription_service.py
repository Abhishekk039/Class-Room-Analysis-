"""
Unified Audio Transcription Service (Stage 1 ASR)

Supports:
    1. Gemini API ASR using the official google-genai SDK.
    2. Gemini-only mode for the deployed Render environment.

The local Faster-Whisper fallback is intentionally disabled because
the Render deployment has limited memory and Whisper can exceed the
512 MB memory limit.

Includes automatic retry handling for temporary Gemini 503 errors.
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

from app.services.transcription import RawSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of retries after the initial Gemini request.
GEMINI_MAX_RETRIES = int(
    os.getenv("GEMINI_MAX_RETRIES", "4")
)

# Initial delay between retries.
GEMINI_RETRY_DELAY_SECONDS = float(
    os.getenv("GEMINI_RETRY_DELAY_SECONDS", "5")
)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def _load_vocabulary_terms(
    custom_vocabulary: Optional[List[str]] = None,
) -> List[str]:
    """
    Merge custom vocabulary with the standard domain vocabulary JSON.
    """

    terms = []

    if custom_vocabulary:
        terms.extend(custom_vocabulary)

    if os.path.exists(VOCABULARY_FILE):
        try:
            with open(
                VOCABULARY_FILE,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            terms.extend(data.get("terms", []))

        except Exception as e:
            logger.warning(
                f"Could not load domain vocabulary from "
                f"{VOCABULARY_FILE}: {e}"
            )

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(terms))


# ---------------------------------------------------------------------------
# Gemini transcription
# ---------------------------------------------------------------------------

def transcribe_audio_gemini(
    audio_path: str,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    custom_vocabulary: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> List[RawSegment]:
    """
    Transcribe audio using Gemini File API.

    Uses GEMINI_MODEL from configuration.

    Temporary 503 / UNAVAILABLE errors are retried using
    exponential backoff.

    Returns:
        List[RawSegment]
    """

    key = (
        api_key
        or GEMINI_API_KEY
        or os.getenv("GEMINI_API_KEY")
    )

    if not key:
        raise ValueError("GEMINI_API_KEY is not set.")

    # -----------------------------------------------------------------------
    # Google GenAI SDK
    # -----------------------------------------------------------------------

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
            raise ImportError(
                "Neither 'google-genai' nor "
                "'google-generativeai' SDK is installed."
            )

    # -----------------------------------------------------------------------
    # Vocabulary
    # -----------------------------------------------------------------------

    terms = _load_vocabulary_terms(custom_vocabulary)

    vocab_str = (
        ", ".join(terms)
        if terms
        else "Standard classroom terminology"
    )

    # -----------------------------------------------------------------------
    # Prompt
    # -----------------------------------------------------------------------

    prompt = f"""
You are an expert audio transcription system specializing in
classroom lectures.

INSTRUCTIONS:

1. Transcribe the spoken audio with precise start and end timestamps
   in seconds.

2. Maintain the original spoken speech, including Indian English
   accents, numbers, technical terms, and code-switched phrases
   such as Hindi/English.

3. Do NOT translate non-English speech into English.

4. Identify speaker roles:
   - "teacher" for explanations and instructions.
   - "student" for answers and short questions.

5. Flag questions accurately using:
   "is_question": true or false.

6. Technical domain vocabulary:
   [{vocab_str}]

   Pay special attention to the spelling of technical terms,
   formulas, programming terminology, electronics terminology,
   and numbers.

7. Do not invent speech.

8. Do not create text for silence, background noise, or unclear audio.

9. Return ONLY a JSON object.

Required format:

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

        # -------------------------------------------------------------------
        # Upload audio
        # -------------------------------------------------------------------

        logger.info(
            f"Uploading audio file {audio_path} "
            f"to Gemini File API..."
        )

        if sdk_type == "genai":

            uploaded_file = client.files.upload(
                file=audio_path
            )

            logger.info(
                f"Gemini file uploaded: {uploaded_file.name}"
            )

            # ---------------------------------------------------------------
            # Wait for Gemini to finish processing the file
            # ---------------------------------------------------------------

            while uploaded_file.state.name == "PROCESSING":

                logger.info(
                    "Gemini is processing the uploaded audio..."
                )

                time.sleep(2)

                uploaded_file = client.files.get(
                    name=uploaded_file.name
                )

            if uploaded_file.state.name == "FAILED":

                error_message = getattr(
                    uploaded_file.error,
                    "message",
                    "Unknown Gemini File API error",
                )

                raise RuntimeError(
                    f"Gemini File API processing failed: "
                    f"{error_message}"
                )

            logger.info(
                "Gemini audio processing completed."
            )

            # ---------------------------------------------------------------
            # Gemini generation with retry
            # ---------------------------------------------------------------

            response = None

            for attempt in range(
                GEMINI_MAX_RETRIES + 1
            ):

                try:

                    logger.info(
                        f"Gemini transcription request "
                        f"attempt {attempt + 1}/"
                        f"{GEMINI_MAX_RETRIES + 1} "
                        f"using model {GEMINI_MODEL}"
                    )

                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=[
                            uploaded_file,
                            prompt,
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.2,
                        ),
                    )

                    # Successful request.
                    break

                except Exception as e:

                    error_text = str(e)

                    # -------------------------------------------------------
                    # Retry temporary 503 errors.
                    # -------------------------------------------------------

                    is_503 = (
                        "503" in error_text
                        or "UNAVAILABLE" in error_text
                        or "high demand" in error_text.lower()
                        or "temporarily unavailable" in error_text.lower()
                    )

                    if not is_503:

                        logger.error(
                            f"Gemini request failed with "
                            f"non-retryable error: {e}"
                        )

                        raise

                    # No attempts remaining.
                    if attempt >= GEMINI_MAX_RETRIES:

                        logger.error(
                            "Gemini remained unavailable after "
                            f"{GEMINI_MAX_RETRIES + 1} attempts."
                        )

                        raise RuntimeError(
                            "Gemini transcription failed after "
                            f"{GEMINI_MAX_RETRIES + 1} attempts. "
                            "The Gemini model is temporarily "
                            "experiencing high demand."
                        ) from e

                    # -------------------------------------------------------
                    # Exponential backoff
                    # -------------------------------------------------------

                    delay = (
                        GEMINI_RETRY_DELAY_SECONDS
                        * (2 ** attempt)
                    )

                    logger.warning(
                        f"Gemini returned 503/UNAVAILABLE. "
                        f"Retrying in {delay:.1f} seconds..."
                    )

                    time.sleep(delay)

            if response is None:
                raise RuntimeError(
                    "Gemini returned no response."
                )

            response_text = response.text

        else:

            # ----------------------------------------------------------------
            # Legacy SDK
            # ----------------------------------------------------------------

            uploaded_file = legacy_genai.upload_file(
                audio_path
            )

            model = legacy_genai.GenerativeModel(
                GEMINI_MODEL
            )

            response = None

            for attempt in range(
                GEMINI_MAX_RETRIES + 1
            ):

                try:

                    logger.info(
                        f"Gemini legacy request "
                        f"attempt {attempt + 1}/"
                        f"{GEMINI_MAX_RETRIES + 1}"
                    )

                    response = model.generate_content(
                        [
                            uploaded_file,
                            prompt,
                        ]
                    )

                    break

                except Exception as e:

                    error_text = str(e)

                    is_503 = (
                        "503" in error_text
                        or "UNAVAILABLE" in error_text
                        or "high demand" in error_text.lower()
                        or "temporarily unavailable" in error_text.lower()
                    )

                    if not is_503:
                        raise

                    if attempt >= GEMINI_MAX_RETRIES:

                        raise RuntimeError(
                            "Gemini transcription failed after "
                            f"{GEMINI_MAX_RETRIES + 1} attempts."
                        ) from e

                    delay = (
                        GEMINI_RETRY_DELAY_SECONDS
                        * (2 ** attempt)
                    )

                    logger.warning(
                        f"Gemini returned 503/UNAVAILABLE. "
                        f"Retrying in {delay:.1f} seconds..."
                    )

                    time.sleep(delay)

            if response is None:
                raise RuntimeError(
                    "Gemini returned no response."
                )

            response_text = response.text

        # -------------------------------------------------------------------
        # Parse JSON
        # -------------------------------------------------------------------

        response_text = response_text.strip()

        # Remove markdown code fences if Gemini adds them.
        if response_text.startswith("```json"):
            response_text = response_text[7:]

        elif response_text.startswith("```"):
            response_text = response_text[3:]

        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        data = json.loads(response_text)

        segments_json = data.get(
            "segments",
            []
        )

        raw_segments: List[RawSegment] = []

        for s in segments_json:

            text = str(
                s.get("text", "")
            ).strip()

            # Ignore empty segments.
            if not text:
                continue

            raw_segments.append(
                RawSegment(
                    start=float(
                        s.get("start", 0.0)
                    ),
                    end=float(
                        s.get("end", 0.0)
                    ),
                    text=text,
                    confidence=float(
                        s.get("confidence", 0.90)
                    ),
                )
            )

        logger.info(
            f"Gemini transcription successful: "
            f"{len(raw_segments)} segments generated."
        )

        return raw_segments

    finally:

        # -------------------------------------------------------------------
        # Cleanup Gemini uploaded file
        # -------------------------------------------------------------------

        if uploaded_file:

            try:

                if sdk_type == "genai":

                    client.files.delete(
                        name=uploaded_file.name
                    )

                else:

                    legacy_genai.delete_file(
                        uploaded_file.name
                    )

                logger.info(
                    "Cleaned up temporary Gemini uploaded file."
                )

            except Exception as cleanup_err:

                logger.warning(
                    f"File cleanup warning: {cleanup_err}"
                )


# ---------------------------------------------------------------------------
# Unified transcription entrypoint
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    custom_vocabulary: Optional[List[str]] = None,
    provider: Optional[str] = None,
) -> List[RawSegment]:
    """
    Unified transcription entrypoint.

    Production deployment uses Gemini only.

    Local Faster-Whisper fallback is intentionally disabled to prevent
    Render memory exhaustion.
    """

    target_provider = (
        provider
        or ASR_PROVIDER
        or "gemini"
    ).lower()

    gemini_key = (
        GEMINI_API_KEY
        or os.getenv("GEMINI_API_KEY")
    )

    # -----------------------------------------------------------------------
    # Gemini
    # -----------------------------------------------------------------------

    if target_provider in (
        "gemini",
        "hybrid",
    ):

        if not gemini_key:

            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        logger.info(
            f"Starting Stage 1 ASR: Gemini "
            f"using {GEMINI_MODEL}"
        )

        try:

            return transcribe_audio_gemini(
                audio_path=audio_path,
                language=language,
                domain=domain,
                custom_vocabulary=custom_vocabulary,
                api_key=gemini_key,
            )

        except Exception as e:

            logger.error(
                f"Gemini transcription failed: {e}"
            )

            # IMPORTANT:
            # Do NOT fall back to Whisper.
            # Render has a 512 MB memory limit and Whisper can
            # cause the application instance to run out of memory.

            raise RuntimeError(
                f"Gemini transcription failed: {e}"
            ) from e

    # -----------------------------------------------------------------------
    # Whisper explicitly requested
    # -----------------------------------------------------------------------

    raise RuntimeError(
        f"Unsupported ASR_PROVIDER='{target_provider}'. "
        "This production deployment requires "
        "ASR_PROVIDER=gemini."
    )