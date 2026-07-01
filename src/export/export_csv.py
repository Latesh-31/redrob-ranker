"""Export ranked candidates to submission CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.utils.constants import OUTPUT_DIR
from src.utils.types import RankedCandidate


def export_csv(
    ranked: list[RankedCandidate],
    path: Path | None = None,
    debug_path: Path | None = None,
) -> Path:
    path = path or OUTPUT_DIR / "submission.csv"
    debug_path = debug_path or OUTPUT_DIR / "top100.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in ranked:
            writer.writerow(
                {
                    "candidate_id": row.candidate_id,
                    "rank": row.rank,
                    "score": f"{row.score:.4f}",
                    "reasoning": row.reasoning,
                }
            )

    debug_payload = [
        {
            "candidate_id": r.candidate_id,
            "rank": r.rank,
            "score": r.score,
            "reasoning": r.reasoning,
            "breakdown": r.breakdown.to_dict(),
        }
        for r in ranked
    ]
    with open(debug_path, "w", encoding="utf-8") as handle:
        json.dump(debug_payload, handle, indent=2)

    return path
