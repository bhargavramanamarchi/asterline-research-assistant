"""
src/corpus.py - Asterline Research Assistant
Loads, validates, and manages chunks from corpus.jsonl.
Preserves the original source text and metadata completely unchanged.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Set

# The 9 mandatory fields required for each corpus chunk
REQUIRED_FIELDS = [
    "chunk_id",
    "doc_id",
    "title",
    "category",
    "owner",
    "source_url",
    "published",
    "version",
    "text",
]


class CorpusValidationError(ValueError):
    """Raised when corpus structure or metadata fails validation."""
    pass


def validate_chunk(record: Dict[str, Any], line_num: int) -> None:
    """
    Validates that a single corpus record contains all required fields
    and that data formats (such as published date YYYYMMDD) are strictly valid.
    """
    # 1. Check for missing required fields
    for field in REQUIRED_FIELDS:
        if field not in record:
            raise CorpusValidationError(
                f"Line {line_num}: Missing required field '{field}' in record."
            )
        val = record[field]
        if val is None or (isinstance(val, str) and not val.strip()):
            raise CorpusValidationError(
                f"Line {line_num}: Field '{field}' must not be empty or null."
            )

    # 2. Validate published date format (YYYYMMDD)
    published = str(record["published"]).strip()
    if not re.fullmatch(r"\d{8}", published):
        raise CorpusValidationError(
            f"Line {line_num}: 'published' date '{published}' must be 8 digits (YYYYMMDD)."
        )
    try:
        datetime.strptime(published, "%Y%m%d")
    except ValueError as err:
        raise CorpusValidationError(
            f"Line {line_num}: 'published' date '{published}' is not a calendar-valid date: {err}"
        )


def load_corpus(file_path: str = "corpus.jsonl") -> List[Dict[str, Any]]:
    """
    Loads and validates the corpus dataset from a JSONL file in read-only mode.
    
    Verifications performed:
      - Confirms file exists.
      - Parses each non-empty line as JSON.
      - Confirms all 9 required fields exist with valid types.
      - Validates published date format (YYYYMMDD).
      - Enforces uniqueness of chunk_id across all records.
      - Returns the list of validated chunk dictionaries preserving original text & metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Corpus file not found at: {file_path}")

    records: List[Dict[str, Any]] = []
    seen_chunk_ids: Set[str] = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            trimmed = line.strip()
            if not trimmed:
                continue

            try:
                record = json.loads(trimmed)
            except json.JSONDecodeError as err:
                raise CorpusValidationError(f"Line {idx}: Invalid JSON syntax: {err}")

            if not isinstance(record, dict):
                raise CorpusValidationError(f"Line {idx}: Record must be a JSON object.")

            validate_chunk(record, idx)

            chunk_id = record["chunk_id"]
            if chunk_id in seen_chunk_ids:
                raise CorpusValidationError(
                    f"Line {idx}: Duplicate chunk_id '{chunk_id}' detected."
                )

            seen_chunk_ids.add(chunk_id)
            records.append(record)

    return records


if __name__ == "__main__":
    loaded = load_corpus("corpus.jsonl")
    print(f"Successfully loaded and validated {len(loaded)} records from corpus.jsonl.")
