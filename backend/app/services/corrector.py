"""
Stage 2 Context-Aware Correction Agent (Gemini API)

Responsibility:
  "Given the audio transcript, classroom context, technical vocabulary,
   and surrounding sentences, are any words clearly ASR errors?"

Strict Constraints:
  - Accuracy and faithfulness are more important than making every sentence grammatically perfect.
  - NEVER allow Stage 2 to fabricate speech or rewrite sentences freely.
  - Preserve exact line order, timestamps, and speaker labels.
"""
import json
import logging
import os
from typing import List, Optional

from app.core.config import ENABLE_STAGE2_CORRECTION, GEMINI_API_KEY, GEMINI_MODEL, VOCABULARY_FILE
from app.models.schemas import TranscriptSegment
from app.services.quality import run_quality_pipeline, RawSegment


logger = logging.getLogger(__name__)


def _load_vocabulary_terms(custom_vocabulary: Optional[List[str]] = None) -> List[str]:
    terms = []
    if custom_vocabulary:
        terms.extend(custom_vocabulary)
    if os.path.exists(VOCABULARY_FILE):
        try:
            with open(VOCABULARY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                terms.extend(data.get("terms", []))
        except Exception:
            pass
    return list(dict.fromkeys(terms))


def run_stage2_correction_gemini(
    segments: List[RawSegment],
    custom_vocabulary: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> List[RawSegment]:
    """
    Passes raw transcript segments to Gemini Stage 2 Contextual Corrector.
    Fixes homophone ASR mistakes, domain terms, and number formats.
    """
    key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not key:
        return segments

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
            logger.warning("Gemini SDK not available for Stage 2 correction.")
            return segments

    vocab_terms = _load_vocabulary_terms(custom_vocabulary)
    vocab_str = ", ".join(vocab_terms) if vocab_terms else "Electronics, Physics, Math"

    input_payload = [
        {
            "id": idx,
            "text": seg.text,
            "start": seg.start,
            "end": seg.end,
        }
        for idx, seg in enumerate(segments)
    ]

    prompt = f"""
    You are Stage 2 Context-Aware Correction Agent for classroom speech recognition.
    
    PRIMARY QUESTION TO ANSWER:
    "Given the audio transcript, classroom context, technical vocabulary, and surrounding sentences, are any words clearly ASR errors?"

    STRICT CONSTRAINTS:
    1. NEVER fabricate spoken words or rewrite sentences freely.
    2. Do NOT convert spoken code-switched phrases or Indian English into artificial standard English.
    3. Correct clear phonetic ASR misspellings, homophones, and domain terms using the target vocabulary context: [{vocab_str}].
       Examples:
       - "home law" -> "Ohm's law"
       - "led circuit" -> "LED circuit"
       - "to hundred volts" -> "200 volts"
       - "re sister" -> "resistor"
    4. Keep line-by-line 1:1 mapping with the input items by "id".
    5. Return ONLY a JSON object formatted as:

    {{
      "corrections": [
        {{
          "id": 0,
          "corrected_text": "corrected string here",
          "correction_applied": true/false
        }}
      ]
    }}

    Input Transcript Data:
    {json.dumps(input_payload, ensure_ascii=False)}
    """

    try:
        if sdk_type == "genai":
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            response_text = response.text
        else:
            model = legacy_genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content([prompt])
            response_text = response.text

        res_data = json.loads(response_text)
        corrections_map = {
            item["id"]: item for item in res_data.get("corrections", [])
        }

        updated_segments = []
        for idx, seg in enumerate(segments):
            corr = corrections_map.get(idx)
            if corr and corr.get("correction_applied") and corr.get("corrected_text"):
                new_text = str(corr["corrected_text"]).strip()
                if new_text and new_text != seg.text:
                    updated_segments.append(
                        RawSegment(
                            start=seg.start,
                            end=seg.end,
                            text=new_text,
                            confidence=seg.confidence,
                            suspicious=seg.suspicious,
                            correction_applied=True,
                            original_text=seg.text,
                        )
                    )
                    continue
            updated_segments.append(seg)

        logger.info(f"Stage 2 correction complete. Applied corrections to {sum(1 for s in updated_segments if s.correction_applied)} segments.")
        return updated_segments

    except Exception as e:
        logger.error(f"Stage 2 Gemini correction agent error ({e}). Returning Stage 1 segments.")
        return segments


def run_stage2_correction(
    segments: List[RawSegment],
    custom_vocabulary: Optional[List[str]] = None,
    provider: Optional[str] = None,
) -> List[RawSegment]:
    """
    Main Stage 2 entrypoint.
    Runs Gemini Stage 2 corrector if enabled and key is present,
    otherwise applies standard rule-based quality filter.
    """
    gemini_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if ENABLE_STAGE2_CORRECTION and gemini_key:
        logger.info("Executing Stage 2: Gemini Context-Aware Correction Agent...")
        corrected = run_stage2_correction_gemini(segments, custom_vocabulary, api_key=gemini_key)
        # Apply conservative deduplication / repetition check on top
        return run_quality_pipeline(corrected)

    logger.info("Stage 2 Gemini key not present; executing rule-based quality pipeline...")
    return run_quality_pipeline(segments)

