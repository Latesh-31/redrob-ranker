"""Shared dataclasses for the ranking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateFeatures:
    candidate_id: str
    skills: list[str] = field(default_factory=list)
    skill_count: int = 0
    total_years: float = 0.0
    relevant_years: float = 0.0
    current_title: str = ""
    title_history: list[str] = field(default_factory=list)
    max_education_level: int = 0
    education_field: str = ""
    cert_count: int = 0
    language_count: int = 0
    profile_completeness: float = 0.5
    honeypot_penalty: float = 0.0
    honeypot_flags: list[str] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)
    embedding_text: str = ""
    summary: str = ""
    career_count: int = 0
    country: str = ""
    last_active_date: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> CandidateFeatures:
        skills = row.get("skills", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        titles = row.get("title_history", [])
        if isinstance(titles, str):
            titles = [t.strip() for t in titles.split("|") if t.strip()]
        flags = row.get("honeypot_flags", [])
        if isinstance(flags, str):
            flags = [f.strip() for f in flags.split("|") if f.strip()]

        signals = {}
        for key, value in row.items():
            if key.startswith("signal_"):
                try:
                    signals[key.replace("signal_", "")] = float(value)
                except (TypeError, ValueError):
                    pass

        return cls(
            candidate_id=str(row["candidate_id"]),
            skills=skills,
            skill_count=int(row.get("skill_count", len(skills))),
            total_years=float(row.get("total_years", 0)),
            relevant_years=float(row.get("relevant_years", 0)),
            current_title=str(row.get("current_title", "")),
            title_history=titles,
            max_education_level=int(row.get("max_education_level", 0)),
            education_field=str(row.get("education_field", "")),
            cert_count=int(row.get("cert_count", 0)),
            language_count=int(row.get("language_count", 0)),
            profile_completeness=float(row.get("profile_completeness", 0.5)),
            honeypot_penalty=float(row.get("honeypot_penalty", 0)),
            honeypot_flags=flags,
            signals=signals,
            embedding_text=str(row.get("embedding_text", "")),
            summary=str(row.get("summary", "")),
            country=str(row.get("country", "")),
            last_active_date=str(row.get("last_active_date", "")),
        )


@dataclass
class JDRequirements:
    title: str = ""
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    min_years: float = 0.0
    max_years: float = 50.0
    seniority: str = "mid"
    required_education_level: int = 1
    preferred_education_level: int = 2
    education_fields: list[str] = field(default_factory=list)
    embedding_text: str = ""
    raw_text: str = ""

    def all_skills(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for skill in self.required_skills + self.preferred_skills:
            norm = skill.lower().strip()
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)
        return result


@dataclass
class ScoreBreakdown:
    candidate_id: str
    semantic: float = 0.0
    skill: float = 0.0
    experience: float = 0.0
    title: float = 0.0
    education: float = 0.0
    behavior: float = 0.0
    consistency: float = 1.0
    honeypot_penalty: float = 0.0
    final_score: float = 0.0
    matched_skills: list[str] = field(default_factory=list)
    honeypot_flags: list[str] = field(default_factory=list)

    def weighted_contributions(self, weights: dict[str, float]) -> list[tuple[str, float]]:
        components = {
            "semantic": self.semantic,
            "skill": self.skill,
            "experience": self.experience,
            "title": self.title,
            "education": self.education,
            "behavior": self.behavior,
        }
        contribs = [(name, weights.get(name, 0) * score) for name, score in components.items()]
        contribs.sort(key=lambda x: x[1], reverse=True)
        return contribs

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "semantic": self.semantic,
            "skill": self.skill,
            "experience": self.experience,
            "title": self.title,
            "education": self.education,
            "behavior": self.behavior,
            "consistency": self.consistency,
            "honeypot_penalty": self.honeypot_penalty,
            "final_score": self.final_score,
            "matched_skills": self.matched_skills,
            "honeypot_flags": self.honeypot_flags,
        }


@dataclass
class RankedCandidate:
    candidate_id: str
    rank: int
    score: float
    reasoning: str
    breakdown: ScoreBreakdown
