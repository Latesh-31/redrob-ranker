"""Export ranked candidates to submission CSV."""

from __future__ import annotations

import json
import pandas as pd
from pathlib import Path

from src.utils.constants import OUTPUT_DIR
from src.utils.types import RankedCandidate


def export_csv(
    ranked: list[RankedCandidate],
    path: Path | None = None,
    debug_path: Path | None = None,
) -> Path:
    if path is None:
        path = OUTPUT_DIR / "submission.xlsx"
    elif path.suffix == ".csv":
        path = path.with_suffix(".xlsx")

    debug_path = debug_path or OUTPUT_DIR / "submission.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "candidate_id": row.candidate_id,
            "rank": row.rank,
            "score": round(row.score, 4),
            "reasoning": row.reasoning,
        }
        for row in ranked
    ]
    df = pd.DataFrame(data)
    df.to_excel(path, index=False)

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
