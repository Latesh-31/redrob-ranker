"""Build structured feature table for ranking."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.clean_data import clean_candidate
from src.preprocessing.extract_text import extract_candidate_text
from src.scoring.honeypot import detect_honeypot
from src.utils.constants import FEATURES_DIR, SIGNAL_NAMES
from src.utils.helpers import write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_feature_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in candidates:
        cleaned = clean_candidate(raw)
        extracted = extract_candidate_text(cleaned)
        penalty, flags = detect_honeypot(cleaned, extracted)
        signals = extracted.get("redrob_signals") or {}

        row: dict[str, Any] = {
            "candidate_id": cleaned["candidate_id"],
            "country": cleaned["profile"]["country"],
            "skills": ",".join(extracted["skills_list"]),
            "skill_count": len(extracted["skills_list"]),
            "total_years": extracted["total_years"],
            "relevant_years": extracted["relevant_years"],
            "current_title": extracted["current_title"],
            "title_history": "|".join(extracted["titles_list"]),
            "max_education_level": extracted["max_education_level"],
            "education_field": extracted["education_field"],
            "cert_count": extracted["cert_count"],
            "language_count": extracted["language_count"],
            "profile_completeness": extracted["profile_completeness"],
            "career_count": extracted["career_count"],
            "summary": extracted["summary"],
            "embedding_text": extracted["embedding_text"],
            "honeypot_penalty": penalty,
            "honeypot_flags": "|".join(flags),
            "last_active_date": str(signals.get("last_active_date", "")),
        }
        for signal in SIGNAL_NAMES:
            row[f"signal_{signal}"] = float(signals.get(signal, np.nan))
        rows.append(row)
    return rows


def normalize_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    for signal in SIGNAL_NAMES:
        col = f"signal_{signal}"
        if col not in df.columns:
            continue
        series = df[col].astype(float)
        valid = series.dropna()
        if valid.empty:
            df[col] = 0.5
            continue
        low, high = valid.quantile(0.01), valid.quantile(0.99)
        if high <= low:
            df[col] = 0.5
            continue
        clipped = series.clip(low, high)
        df[col] = ((clipped - low) / (high - low)).fillna(0.5)
    return df


def build_skill_vocab(skills_lists: list[list[str]], top_n: int = 5000) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for skills in skills_lists:
        counter.update(skills)
    return dict(counter.most_common(top_n))


def save_features(rows: list[dict[str, Any]], output_dir: FEATURES_DIR | None = None) -> pd.DataFrame:
    output_dir = output_dir or FEATURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df = normalize_signal_columns(df)

    skills_lists = [[s for s in str(row).split(",") if s] for row in df["skills"].tolist()]
    vocab = build_skill_vocab(skills_lists)
    write_json(output_dir / "skill_vocab.json", vocab)

    parquet_path = output_dir / "structured_features.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved structured features (%d rows) to %s", len(df), parquet_path)
    return df
