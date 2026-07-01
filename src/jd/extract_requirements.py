"""Extract structured requirements from a parsed job description."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.jd.parse_jd import parse_job_description
from src.utils.constants import FEATURES_DIR, SENIORITY_KEYWORDS
from src.utils.helpers import normalize_skill, read_json, tokenize
from src.utils.types import JDRequirements

TECH_SKILLS = [
    "python",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "machine learning",
    "deep learning",
    "nlp",
    "embeddings",
    "ranking",
    "retrieval",
    "sql",
    "aws",
    "gcp",
    "azure",
    "docker",
    "kubernetes",
    "spark",
    "java",
    "javascript",
    "typescript",
    "react",
    "node",
    "postgresql",
    "mongodb",
    "redis",
    "kafka",
    "airflow",
    "mlops",
    "vector search",
    "faiss",
    "sentence-transformers",
    "recommendation systems",
    "feature engineering",
    "statistics",
    "pandas",
    "numpy",
]

DEGREE_PATTERNS = [
    (r"\bph\.?d\b", 4),
    (r"\bdoctorate\b", 4),
    (r"\bm\.?s\.?\b|\bmasters?\b", 3),
    (r"\bm\.?b\.?a\.?\b", 3),
    (r"\bb\.?s\.?\b|\bbachelors?\b", 2),
    (r"\bassociate\b", 1),
]


def _load_skill_vocab() -> set[str]:
    vocab_path = FEATURES_DIR / "skill_vocab.json"
    if vocab_path.exists():
        vocab = read_json(vocab_path)
        return set(vocab.keys())
    return set(TECH_SKILLS)


def _extract_years(text: str) -> float:
    text_lower = text.lower()
    years: list[int] = []
    years.extend(int(m) for m in re.findall(r"(\d+)\+\s*years?", text_lower))
    patterns = [
        r"(\d+)\s*(?:years?|yrs?)\s*(?:of\s+)?(?:professional|relevant|work)?\s*[\w\s]{0,40}experience",
        r"(\d+)\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)",
        r"at least\s+(\d+)\s+years?",
        r"minimum\s+(\d+)\s+years?",
    ]
    for pattern in patterns:
        years.extend(int(m) for m in re.findall(pattern, text_lower))
    return float(max(years)) if years else 0.0


def _extract_seniority(text: str, title: str) -> str:
    combined = f"{title} {text}".lower()
    for keyword, level in SENIORITY_KEYWORDS.items():
        if keyword in combined:
            return level
    return "mid"


def _extract_skills(text: str, vocab: set[str]) -> list[str]:
    text_lower = text.lower()
    found: list[str] = []
    seen: set[str] = set()

    for skill in sorted(vocab | set(TECH_SKILLS), key=len, reverse=True):
        norm = normalize_skill(skill)
        if norm in seen:
            continue
        pattern = re.escape(skill).replace(r"\ ", r"[\s-]?")
        if re.search(pattern, text_lower):
            seen.add(norm)
            found.append(norm)

    bold_matches = re.findall(r"\*\*([^*]+)\*\*", text)
    for match in bold_matches:
        norm = normalize_skill(match)
        if norm and norm not in seen:
            seen.add(norm)
            found.append(norm)

    return found


def _extract_education_level(text: str) -> tuple[int, list[str]]:
    text_lower = text.lower()
    level = 0
    for pattern, value in DEGREE_PATTERNS:
        if re.search(pattern, text_lower):
            level = max(level, value)

    fields: list[str] = []
    field_patterns = [
        r"computer science",
        r"statistics",
        r"engineering",
        r"mathematics",
        r"data science",
    ]
    for fp in field_patterns:
        if re.search(fp, text_lower):
            fields.append(fp.replace(" ", " "))
    return level, fields


def extract_requirements(parsed_jd: dict[str, Any] | None = None, jd_path: Path | None = None) -> JDRequirements:
    parsed = parsed_jd or parse_job_description(jd_path)
    vocab = _load_skill_vocab()

    required_text = parsed.get("required_text", "")
    preferred_text = parsed.get("preferred_text", "")
    full_text = parsed.get("full_text", parsed.get("raw_text", ""))
    title = parsed.get("title", "")

    required_skills = _extract_skills(required_text, vocab)
    preferred_skills = _extract_skills(preferred_text, vocab)
    if not required_skills:
        required_skills = _extract_skills(full_text, vocab)

    preferred_set = set(preferred_skills)
    required_skills = [s for s in required_skills if s not in preferred_set or s in required_skills[:5]]
    preferred_skills = [s for s in preferred_skills if s not in required_skills]

    min_years = _extract_years(required_text) or _extract_years(full_text)
    seniority = _extract_seniority(full_text, title)
    edu_level, edu_fields = _extract_education_level(required_text or full_text)

    embedding_text = "\n".join(
        part
        for part in [
            f"Title: {title}",
            parsed.get("overview", ""),
            required_text,
            preferred_text,
        ]
        if part
    )

    return JDRequirements(
        title=title,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_years=min_years,
        max_years=min_years + 15 if min_years else 50.0,
        seniority=seniority,
        required_education_level=max(edu_level, 2 if "bs" in full_text.lower() or "b.s" in full_text.lower() else 0),
        preferred_education_level=max(edu_level, 3),
        education_fields=edu_fields,
        embedding_text=embedding_text,
        raw_text=full_text,
    )
