# Vocabulary Files

Each JSON file in this directory defines domain-specific vocabulary for one classroom subject.

## Format

```json
{
  "subject": "electronics",
  "description": "Short description",
  "initial_prompt": "Full sentence used as Whisper initial_prompt",
  "terms": ["list", "of", "key", "terms"],
  "common_corrections": {
    "misheard phrase": "correct term"
  }
}
```

## Switching Subject

Set the `VOCABULARY_FILE` environment variable:

```bash
VOCABULARY_FILE=app/vocabularies/biology.json uvicorn app.main:app
```

Or edit `VOCABULARY_FILE` in `app/core/config.py`.

## Available Vocabularies

| File | Subject |
|---|---|
| `electronics.json` | Circuits, LEDs, resistors, voltage |

## Adding a New Subject

1. Copy `electronics.json` → `your_subject.json`
2. Update `subject`, `description`, `initial_prompt`, `terms`, `common_corrections`
3. Set `VOCABULARY_FILE=app/vocabularies/your_subject.json`
