"""Extract searchable text from candidate profiles."""

from __future__ import annotations

from typing import Any

from src.preprocessing.clean_data import compute_career_years
from src.utils.constants import MAX_EMBEDDING_CHARS
from src.utils.helpers import education_level, normalize_whitespace, truncate_text


def _recent_roles(career_history: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
    return career_history[:limit] if career_history else []


def extract_candidate_text(candidate: dict[str, Any]) -> dict[str, Any]:
    """Build embedding text and structured fields from a cleaned candidate."""
    profile = candidate.get("profile") or {}
    career = candidate.get("career_history") or []
    education = candidate.get("education") or []
    skills = candidate.get("skills") or []
    certs = candidate.get("certifications") or []

    current_title = profile.get("current_title", "")
    summary = profile.get("summary", "")
    titles = [current_title] if current_title else []
    for role in career:
        title = role.get("title", "")
        if title and title not in titles:
            titles.append(title)

    parts: list[str] = []
    if current_title:
        parts.append(f"Title: {current_title}")
    if summary:
        parts.append(f"Summary: {truncate_text(summary, 300)}")
    for role in _recent_roles(career):
        role_bits = [role.get("title", ""), role.get("company", ""), truncate_text(role.get("description", ""), 300)]
        role_text = " | ".join(b for b in role_bits if b)
        if role_text:
            parts.append(f"Experience: {role_text}")
    if skills:
        parts.append("Skills: " + ", ".join(skills))
    if certs:
        parts.append("Certifications: " + ", ".join(certs))

    embedding_text = truncate_text("\n".join(parts), MAX_EMBEDDING_CHARS)

    max_edu_level = 0
    edu_field = ""
    for entry in education:
        level = education_level(entry.get("degree", ""))
        max_edu_level = max(max_edu_level, level)
        if entry.get("field") and not edu_field:
            edu_field = entry.get("field", "")

    total_years = compute_career_years(career)
    signals = candidate.get("redrob_signals") or {}

    return {
        "embedding_text": embedding_text,
        "skills_list": list(skills),
        "titles_list": titles,
        "current_title": current_title,
        "summary": summary,
        "total_years": total_years,
        "relevant_years": total_years,
        "max_education_level": max_edu_level,
        "education_field": edu_field,
        "cert_count": len(certs),
        "language_count": len(candidate.get("languages") or []),
        "career_count": len(career),
        "profile_completeness": float(signals.get("profile_completeness", _estimate_completeness(candidate))),
        "redrob_signals": signals,
    }


def _estimate_completeness(candidate: dict[str, Any]) -> float:
    profile = candidate.get("profile") or {}
    checks = [
        bool(profile.get("summary")),
        bool(profile.get("current_title")),
        bool(candidate.get("career_history")),
        bool(candidate.get("education")),
        bool(candidate.get("skills")),
        bool(candidate.get("certifications")),
    ]
    return round(sum(checks) / len(checks), 2)
