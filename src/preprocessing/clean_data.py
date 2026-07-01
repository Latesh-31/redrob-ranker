"""Clean and normalize raw candidate records."""

from __future__ import annotations

import copy
from typing import Any

from src.utils.helpers import normalize_skills, normalize_whitespace, parse_year_month


def _clean_career_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": normalize_whitespace(str(entry.get("title", ""))),
        "company": normalize_whitespace(str(entry.get("company", ""))),
        "start_date": normalize_whitespace(str(entry.get("start_date", ""))),
        "end_date": normalize_whitespace(str(entry.get("end_date", ""))),
        "description": normalize_whitespace(str(entry.get("description", ""))),
    }


def _clean_education_entry(entry: dict[str, Any]) -> dict[str, Any]:
    degree = entry.get("degree") or ""
    field = entry.get("field_of_study") or entry.get("field") or ""
    school = entry.get("institution") or entry.get("school") or ""
    year = entry.get("end_year") or entry.get("year") or ""

    try:
        year_val = int(year) if year not in ("", None) else None
    except (TypeError, ValueError):
        year_val = None
    return {
        "degree": normalize_whitespace(str(degree)),
        "field": normalize_whitespace(str(field)),
        "school": normalize_whitespace(str(school)),
        "year": year_val,
    }


def clean_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a cleaned copy of a candidate record."""
    cleaned = copy.deepcopy(candidate)
    cleaned["candidate_id"] = str(candidate.get("candidate_id", "")).strip()

    profile = candidate.get("profile") or {}
    name = profile.get("anonymized_name") or profile.get("name") or ""
    cleaned["profile"] = {
        "name": normalize_whitespace(str(name)),
        "summary": normalize_whitespace(str(profile.get("summary", ""))),
        "location": normalize_whitespace(str(profile.get("location", ""))),
        "current_title": normalize_whitespace(str(profile.get("current_title", ""))),
        "country": normalize_whitespace(str(profile.get("country", ""))),
    }

    career = candidate.get("career_history") or []
    cleaned["career_history"] = [_clean_career_entry(e) for e in career if isinstance(e, dict)]

    education = candidate.get("education") or []
    cleaned["education"] = [_clean_education_entry(e) for e in education if isinstance(e, dict)]

    # Extract skill names if list of dicts or list of strings
    raw_skills = candidate.get("skills") or []
    skill_names = []
    if raw_skills:
        if isinstance(raw_skills[0], dict):
            skill_names = [str(s.get("name", "")) for s in raw_skills if isinstance(s, dict)]
        else:
            skill_names = [str(s) for s in raw_skills]
    cleaned["skills"] = normalize_skills(skill_names)

    cleaned["certifications"] = []
    for c in (candidate.get("certifications") or []):
        val = c.get("name", "") if isinstance(c, dict) else c
        if val and str(val).strip():
            cleaned["certifications"].append(normalize_whitespace(str(val)))

    cleaned["languages"] = []
    for lang in (candidate.get("languages") or []):
        val = lang.get("language", "") if isinstance(lang, dict) else lang
        if val and str(val).strip():
            cleaned["languages"].append(normalize_whitespace(str(val)))

    signals = candidate.get("redrob_signals") or {}
    cleaned_signals = {}
    for key, value in signals.items():
        try:
            if isinstance(value, (int, float)):
                cleaned_signals[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    if "last_active_date" in signals:
        cleaned_signals["last_active_date"] = str(signals["last_active_date"])

    # Map new schema signal keys to original schema keys
    if "profile_views_received_30d" in cleaned_signals:
        cleaned_signals["profile_views"] = cleaned_signals["profile_views_received_30d"]
    if "recruiter_response_rate" in cleaned_signals:
        cleaned_signals["response_rate"] = cleaned_signals["recruiter_response_rate"]
    if "profile_completeness_score" in cleaned_signals:
        cleaned_signals["profile_completeness"] = cleaned_signals["profile_completeness_score"] / 100.0

    total_years = compute_career_years(cleaned["career_history"])
    career_count = len(cleaned["career_history"])

    if "avg_tenure_months" not in cleaned_signals:
        if career_count > 0:
            cleaned_signals["avg_tenure_months"] = (total_years * 12.0) / career_count
        else:
            cleaned_signals["avg_tenure_months"] = 0.0

    if "job_hop_score" not in cleaned_signals:
        if total_years > 0.5:
            ratio = career_count / total_years
            cleaned_signals["job_hop_score"] = min(1.0, max(0.0, (ratio - 0.2) / 1.3))
        else:
            cleaned_signals["job_hop_score"] = 0.0

    if "engagement_score" not in cleaned_signals:
        github = cleaned_signals.get("github_activity_score", -1.0)
        github_norm = github / 100.0 if github >= 0 else 0.5
        interview_rate = cleaned_signals.get("interview_completion_rate", 0.5)
        cleaned_signals["engagement_score"] = 0.5 * github_norm + 0.5 * interview_rate

    cleaned["redrob_signals"] = cleaned_signals

    return cleaned


def compute_career_years(career_history: list[dict[str, Any]]) -> float:
    total_months = 0
    for role in career_history:
        start = parse_year_month(role.get("start_date"))
        end = parse_year_month(role.get("end_date"))
        if start and end:
            months = (end.year - start.year) * 12 + (end.month - start.month)
            total_months += max(0, months)
    return round(total_months / 12.0, 2)
