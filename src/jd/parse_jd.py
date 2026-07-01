"""Parse job description markdown into structured sections."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.utils.constants import DATA_DIR
from src.utils.helpers import read_text


def parse_job_description(path: Path | None = None) -> dict[str, Any]:
    path = path or DATA_DIR / "job_description.md"
    text = read_text(path)

    title = ""
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    sections: dict[str, str] = {}
    current_key = "overview"
    current_lines: list[str] = []

    for line in text.splitlines():
        header_match = re.match(r"^##+\s+(.+)$", line)
        if header_match:
            sections[current_key] = "\n".join(current_lines).strip()
            current_key = _normalize_section(header_match.group(1))
            current_lines = []
        else:
            current_lines.append(line)
    sections[current_key] = "\n".join(current_lines).strip()

    required_text = _merge_sections(sections, ["required qualifications", "requirements", "required"])
    preferred_text = _merge_sections(sections, ["preferred qualifications", "preferred", "nice to have"])
    responsibilities = _merge_sections(sections, ["responsibilities", "what you will do"])

    return {
        "title": title,
        "raw_text": text,
        "overview": sections.get("about the role", sections.get("overview", "")),
        "required_text": required_text,
        "preferred_text": preferred_text,
        "responsibilities": responsibilities,
        "full_text": text,
    }


def _normalize_section(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _merge_sections(sections: dict[str, str], keys: list[str]) -> str:
    parts = [sections[k] for k in keys if k in sections and sections[k]]
    return "\n".join(parts)
