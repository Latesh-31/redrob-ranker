"""Load candidate records from JSONL or JSON sources."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator

from src.utils.constants import DATA_DIR, REQUIRED_CANDIDATE_KEYS
from src.utils.helpers import read_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _open_text(path: Path):
    if path.suffix == ".gz" or str(path).endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _validate_candidate(candidate: dict, line_no: int) -> bool:
    for key in REQUIRED_CANDIDATE_KEYS:
        if key not in candidate:
            logger.warning("Line %d: missing required key %r", line_no, key)
            return False
    return True


def load_candidates(path: Path | None = None) -> Iterator[dict]:
    """Yield candidate records from JSONL, gzipped JSONL, or JSON array."""
    path = path or DATA_DIR / "candidates.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    if path.suffix == ".json" and not str(path).endswith(".jsonl"):
        records = read_json(path)
        if not isinstance(records, list):
            raise ValueError(f"Expected JSON array in {path}")
        for idx, candidate in enumerate(records, start=1):
            if isinstance(candidate, dict) and _validate_candidate(candidate, idx):
                yield candidate
        return

    with _open_text(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d: invalid JSON (%s)", line_no, exc)
                continue
            if not isinstance(candidate, dict):
                logger.warning("Line %d: expected object, got %s", line_no, type(candidate))
                continue
            if _validate_candidate(candidate, line_no):
                yield candidate


def load_candidates_list(path: Path | None = None) -> list[dict]:
    return list(load_candidates(path))
