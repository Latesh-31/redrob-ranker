"""Honeypot detection rules."""

from __future__ import annotations

from typing import Any

from src.utils.constants import HONEYPOT_SKILL_BLOCKLIST
from src.utils.helpers import jaccard, normalize_whitespace, tokenize


def detect_honeypot(candidate: dict[str, Any], extracted: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Return (penalty in [0, 1], list of triggered rule names).
    Penalties stack with diminishing returns via min(1.0, sum * 0.85).
    """
    penalties: list[tuple[str, float]] = []

    skills = extracted.get("skills_list") or candidate.get("skills") or []
    skill_count = len(skills)
    embedding_text = extracted.get("embedding_text", "")
    career = candidate.get("career_history") or []
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}

    if skill_count > 80:
        penalties.append(("skill_stuffing", 0.45))
    elif skill_count > 50:
        penalties.append(("skill_stuffing", 0.25))

    if embedding_text and skills:
        skills_blob = " ".join(skills)
        if len(skills_blob) > len(embedding_text) * 0.5 and skill_count > 30:
            penalties.append(("skills_dominate_text", 0.3))

    if not career and len(normalize_whitespace(profile.get("summary", ""))) > 100:
        penalties.append(("empty_shell", 0.5))

    summary = profile.get("summary", "")
    if summary and career:
        for role in career:
            desc = role.get("description", "")
            if desc and jaccard(tokenize(summary), tokenize(desc)) > 0.85:
                penalties.append(("duplicate_content", 0.35))
                break

    inflated_titles = {"ceo", "founder", "president", "chief"}
    title_tokens = tokenize(profile.get("current_title", ""))
    if title_tokens & inflated_titles and not career:
        penalties.append(("title_inflation", 0.35))

    skill_set = {s.lower() for s in skills}
    if skill_count >= 15 and HONEYPOT_SKILL_BLOCKLIST.issubset(skill_set):
        penalties.append(("blocklist_skills", 1.0))

    buzzwords = ["python", "pytorch", "tensorflow", "machine learning", "nlp", "embeddings"]
    summary_lower = summary.lower()
    buzz_count = sum(summary_lower.count(w) for w in buzzwords)
    if buzz_count > 12 and len(career) <= 1:
        penalties.append(("keyword_echo", 0.35))

    completeness = float(signals.get("profile_completeness", extracted.get("profile_completeness", 0.5)))
    perfect_signals = all(float(signals.get(k, 0)) >= 0.99 for k in ("response_rate", "engagement_score") if k in signals)
    if perfect_signals and completeness < 0.3:
        penalties.append(("signal_anomaly", 0.25))

    # 1. Consulting firms only
    if career:
        consulting_keywords = {"tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tech mahindra", "hcl"}
        only_consulting = True
        for role in career:
            company = str(role.get("company", "")).lower()
            if not any(k in company for k in consulting_keywords):
                only_consulting = False
                break
        if only_consulting:
            penalties.append(("only_consulting_firms", 0.5))

    # 2. Title chaser
    career_count = len(career)
    total_years = extracted.get("total_years", 0.0)
    avg_tenure = (total_years * 12.0) / career_count if career_count > 0 else 0.0
    if career_count >= 3 and 0.0 < avg_tenure < 18.0:
        has_promotions = False
        for role in career:
            role_title = str(role.get("title", "")).lower()
            if any(k in role_title for k in ("senior", "lead", "staff", "principal", "director")):
                has_promotions = True
                break
        if has_promotions:
            penalties.append(("title_chaser", 0.35))

    # 3. Pure research / academic only (no production)
    if career:
        all_research = True
        for role in career:
            role_title = str(role.get("title", "")).lower()
            is_research = any(k in role_title for k in ("research", "scientist", "postdoc", "phd", "academic", "fellow", "student", "intern"))
            is_eng = any(k in role_title for k in ("engineer", "developer", "architect", "programmer", "manager", "lead", "director"))
            if not is_research or is_eng:
                all_research = False
                break
        if all_research:
            penalties.append(("pure_research", 0.45))

    # 4. Location mismatch (outside India, not willing to relocate)
    country = str(profile.get("country", "")).lower().strip()
    willing_to_relocate = float(signals.get("willing_to_relocate", 1.0))
    if country and country != "india" and not willing_to_relocate:
        penalties.append(("location_mismatch", 0.4))

    # 5. Inactive & unresponsive candidates (not logged in for 6 months and response rate <= 0.05)
    last_active = signals.get("last_active_date", "")
    response_rate = float(signals.get("response_rate", 1.0))
    if last_active:
        from datetime import datetime
        try:
            active_date = datetime.strptime(last_active, "%Y-%m-%d")
            ref_date = datetime(2026, 6, 30)
            days_inactive = (ref_date - active_date).days
            if days_inactive >= 180 and response_rate <= 0.05:
                penalties.append(("inactive_unresponsive", 0.6))
        except Exception:
            pass

    if not penalties:
        return 0.0, []

    total = sum(p for _, p in penalties)
    penalty = min(1.0, total * 0.85)
    flags = [name for name, _ in penalties]
    return round(penalty, 4), flags
