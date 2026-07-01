"""Validate submission CSV format and constraints."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"
REQUIRED_COLUMNS = ("candidate_id", "rank", "score", "reasoning")
EXPECTED_ROWS = 100


def validate_submission(path: Path = SUBMISSION_FILE, expected_rows: int = EXPECTED_ROWS) -> bool:
    if not path.exists():
        print(f"Error: submission file not found at {path}")
        return False

    errors: list[str] = []
    rows: list[dict[str, str]] = []

    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            errors.append("CSV has no header row")
        else:
            missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
            if missing:
                errors.append(f"Missing columns: {missing}")
        rows = list(reader)

    if len(rows) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, got {len(rows)}")

    ranks: list[int] = []
    ids: set[str] = set()

    for i, row in enumerate(rows, start=2):
        cid = row.get("candidate_id", "").strip()
        if not cid:
            errors.append(f"Row {i}: empty candidate_id")
        elif cid in ids:
            errors.append(f"Row {i}: duplicate candidate_id {cid}")
        else:
            ids.add(cid)

        try:
            rank = int(row.get("rank", ""))
            ranks.append(rank)
        except ValueError:
            errors.append(f"Row {i}: invalid rank {row.get('rank')}")

        try:
            score = float(row.get("score", ""))
            if not (0.0 <= score <= 1.0):
                errors.append(f"Row {i}: score {score} out of [0, 1] range")
        except ValueError:
            errors.append(f"Row {i}: invalid score {row.get('score')}")

        reasoning = row.get("reasoning", "").strip()
        if not reasoning:
            errors.append(f"Row {i}: empty reasoning")

    if ranks:
        expected = list(range(1, expected_rows + 1))
        if sorted(ranks) != expected:
            errors.append(f"Ranks must be unique integers 1..{expected_rows}")

    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"  - {err}")
        return False

    print(f"Validation passed: {path} ({len(rows)} rows)")
    return True


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SUBMISSION_FILE
    success = validate_submission(path)
    sys.exit(0 if success else 1)
