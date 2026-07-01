"""Unit tests for preprocessing and scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.clean_data import clean_candidate
from src.preprocessing.extract_text import extract_candidate_text
from src.preprocessing.load_candidates import load_candidates_list
from src.scoring.final_score import compute_breakdown
from src.scoring.honeypot import detect_honeypot
from src.reranking.xgb_reranker import XGBReranker
from src.jd.extract_requirements import extract_requirements
from src.jd.parse_jd import parse_job_description
from src.utils.constants import DATA_DIR
from src.utils.types import CandidateFeatures, JDRequirements

SAMPLE_PATH = DATA_DIR / "sample_candidates.json"


@pytest.fixture
def sample_candidates() -> list[dict]:
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_load_sample_candidates(sample_candidates: list[dict]) -> None:
    assert len(sample_candidates) >= 2
    assert all("candidate_id" in c for c in sample_candidates)


def test_clean_lowercases_skills(sample_candidates: list[dict]) -> None:
    cleaned = clean_candidate(sample_candidates[0])
    raw_skills = sample_candidates[0]["skills"]
    raw_names = [s.get("name") if isinstance(s, dict) else s for s in raw_skills]
    assert cleaned["skills"] == [s.lower().replace(" ", "-") if " " in s else s.lower() for s in raw_names]


def test_extract_text_non_empty(sample_candidates: list[dict]) -> None:
    cleaned = clean_candidate(sample_candidates[0])
    extracted = extract_candidate_text(cleaned)
    assert extracted["embedding_text"]
    assert len(extracted["embedding_text"]) > 20


def test_honeypot_detected() -> None:
    # Construct a mock honeypot candidate
    honeypot = {
        "candidate_id": "CAND_HONEYPOT",
        "profile": {
            "name": "Super Candidate",
            "summary": "Python PyTorch TensorFlow scikit-learn NLP embeddings ranking SQL AWS Python PyTorch TensorFlow machine learning expert.",
            "location": "Remote",
            "current_title": "CEO Founder ML Architect"
        },
        "career_history": [],
        "education": [
            {"degree": "High School", "field": "General", "school": "Local High", "year": 2020}
        ],
        "skills": [{"name": s, "proficiency": "expert", "endorsements": 100} for s in [
            "python", "pytorch", "tensorflow", "scikit-learn", "nlp", "embeddings", "ranking", "sql", "aws",
            "kubernetes", "docker", "spark", "machine learning", "deep learning"
        ]],
        "certifications": [],
        "languages": []
    }
    cleaned = clean_candidate(honeypot)
    extracted = extract_candidate_text(cleaned)
    penalty, flags = detect_honeypot(cleaned, extracted)
    assert penalty >= 0.5
    assert flags


def test_jd_extracts_skills() -> None:
    parsed = parse_job_description(DATA_DIR / "job_description.md")
    req = extract_requirements(parsed)
    assert "python" in req.required_skills or "python" in req.all_skills()
    assert req.min_years >= 5


def test_skill_score_perfect_match() -> None:
    features = CandidateFeatures(
        candidate_id="CAND_PERFECT",
        skills=["python", "pytorch", "tensorflow", "scikit-learn", "nlp"],
        relevant_years=6.0,
        current_title="Senior Machine Learning Engineer",
        title_history=["Senior Machine Learning Engineer"],
        max_education_level=3,
        education_field="Computer Science",
    )
    req = JDRequirements(
        required_skills=["python", "pytorch", "tensorflow", "scikit-learn", "nlp"],
        min_years=5,
        title="Senior Machine Learning Engineer",
    )
    breakdown = compute_breakdown(features, req, semantic_score=0.8)
    assert breakdown.skill >= 0.7
    assert breakdown.final_score > 0.5


def test_load_candidates_from_json(sample_candidates: list[dict], tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(sample_candidates), encoding="utf-8")
    loaded = load_candidates_list(path)
    assert len(loaded) == len(sample_candidates)


def test_xgb_reranker_filters_and_orders_candidates() -> None:
    req = JDRequirements(
        required_skills=["python", "machine learning"],
        preferred_skills=["pytorch"],
        min_years=5,
        title="Machine Learning Engineer",
    )
    strong = CandidateFeatures(
        candidate_id="STRONG",
        skills=["python", "machine learning", "pytorch"],
        skill_count=3,
        total_years=7.0,
        relevant_years=6.0,
        current_title="Senior Machine Learning Engineer",
        title_history=["Machine Learning Engineer"],
        max_education_level=3,
        education_field="Computer Science",
        cert_count=2,
        language_count=1,
        profile_completeness=0.95,
        career_count=4,
        signals={"response_rate": 0.8, "engagement_score": 0.9, "avg_tenure_months": 40.0, "job_hop_score": 0.1},
    )
    weak = CandidateFeatures(
        candidate_id="WEAK",
        skills=["sales", "marketing"],
        skill_count=2,
        total_years=1.0,
        relevant_years=0.0,
        current_title="Account Executive",
        title_history=["Sales Representative"],
        max_education_level=1,
        education_field="Business",
        cert_count=0,
        language_count=1,
        profile_completeness=0.4,
        career_count=1,
        signals={"response_rate": 0.2, "engagement_score": 0.1, "avg_tenure_months": 8.0, "job_hop_score": 0.8},
    )

    strong_breakdown = compute_breakdown(strong, req, semantic_score=0.9)
    weak_breakdown = compute_breakdown(weak, req, semantic_score=0.1)

    reranker = XGBReranker(filter_threshold=0.4)
    records = [
        (strong, strong_breakdown, 0.9),
        (weak, weak_breakdown, 0.1),
    ]

    reranker.fit(records)
    ranked = reranker.rerank(records, minimum_keep=1)

    assert ranked
    assert ranked[0].candidate.candidate_id == "STRONG"
    assert all(0.0 <= item.fit_probability <= 1.0 for item in ranked)
