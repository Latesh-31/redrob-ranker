"""Shared helper utilities."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.constants import DEGREE_LEVELS, SKILL_ALIASES

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
SKILL_SPLIT_RE = re.compile(r"[,;/|]+")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_text(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def strip_html(text: str) -> str:
    return HTML_TAG_RE.sub(" ", text or "")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", strip_html(text)).strip()


def normalize_skill(skill: str) -> str:
    cleaned = normalize_whitespace(skill.lower())
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    for alias, canonical in SKILL_ALIASES.items():
        if cleaned == alias:
            return canonical
    return cleaned.replace(" ", "-") if " " in cleaned else cleaned


def normalize_skills(skills: list[str] | None) -> list[str]:
    if not skills:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for raw in skills:
        norm = normalize_skill(str(raw))
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def parse_year_month(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    text = str(date_str).strip().lower()
    if text in {"present", "current", "now", ""}:
        return datetime.now().replace(day=1)
    for fmt in ("%Y-%m", "%Y/%m", "%Y-%m-%d", "%Y"):
        try:
            parsed = datetime.strptime(text[: len(fmt.replace("%", "0"))], fmt)
            return parsed.replace(day=1)
        except ValueError:
            continue
    match = re.match(r"(\d{4})", text)
    if match:
        return datetime(int(match.group(1)), 1, 1)
    return None


def months_between(start: datetime, end: datetime) -> int:
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))


def education_level(degree: str | None) -> int:
    if not degree:
        return 0
    text = degree.lower().strip()
    for key, level in DEGREE_LEVELS.items():
        if key in text:
            return level
    return 1


def tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9+#]+", text.lower()) if len(t) > 1}


def fuzzy_skill_match(candidate_skill: str, required_skill: str) -> bool:
    if candidate_skill == required_skill:
        return True
    c_tokens = tokenize(candidate_skill.replace("-", " "))
    r_tokens = tokenize(required_skill.replace("-", " "))
    if not c_tokens or not r_tokens:
        return False
    return bool(c_tokens & r_tokens) or candidate_skill in required_skill or required_skill in candidate_skill


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rsplit(" ", 1)[0] + "..."
