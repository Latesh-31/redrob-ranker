"""Generate synthetic candidate dataset for development and testing."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.constants import DATA_DIR

ML_SKILLS = [
    "python", "pytorch", "tensorflow", "scikit-learn", "nlp", "embeddings",
    "ranking", "sql", "aws", "machine learning", "deep learning", "pandas",
    "numpy", "docker", "kubernetes", "vector search", "feature engineering",
]

OTHER_SKILLS = [
    "java", "spring", "react", "angular", "node", "postgresql", "mongodb",
    "sales", "marketing", "excel", "hr", "recruiting",
]

TITLES_ML = [
    "Senior Machine Learning Engineer",
    "Machine Learning Engineer",
    "ML Engineer",
    "Data Scientist",
    "NLP Engineer",
    "Applied Scientist",
]

TITLES_OTHER = [
    "Software Engineer",
    "Backend Developer",
    "Frontend Developer",
    "Product Manager",
    "Sales Representative",
]

SCHOOLS = ["MIT", "Stanford", "Berkeley", "CMU", "Georgia Tech", "UT Austin", "State University"]


def _make_ml_candidate(cid: str, rng: random.Random) -> dict:
    years = rng.randint(3, 12)
    skills = rng.sample(ML_SKILLS, k=rng.randint(6, 12))
    title = rng.choice(TITLES_ML)
    return {
        "candidate_id": cid,
        "profile": {
            "name": f"Candidate {cid}",
            "summary": f"{title} with {years} years building ML and NLP systems in production.",
            "location": rng.choice(["Remote", "San Francisco", "New York", "Seattle"]),
            "current_title": title,
        },
        "career_history": [
            {
                "title": title,
                "company": rng.choice(["MatchTech", "AI Labs", "DataCorp", "SearchCo"]),
                "start_date": f"{2024 - years}-03",
                "end_date": "present",
                "description": "Built ranking pipelines, embeddings, and feature engineering for search.",
            },
            {
                "title": "Machine Learning Engineer" if "Senior" in title else "Data Scientist",
                "company": "PreviousCo",
                "start_date": f"{2024 - years - 3}-01",
                "end_date": f"{2024 - years}-02",
                "description": "Developed PyTorch models and scikit-learn pipelines.",
            },
        ],
        "education": [
            {
                "degree": rng.choice(["MS", "BS", "PhD"]),
                "field": "Computer Science",
                "school": rng.choice(SCHOOLS),
                "year": 2024 - years - 2,
            }
        ],
        "skills": skills,
        "certifications": rng.choice([[], ["AWS Machine Learning Specialty"], ["GCP Professional ML Engineer"]]),
        "languages": ["English"],
        "redrob_signals": {
            "profile_views": rng.randint(50, 800),
            "response_rate": round(rng.uniform(0.5, 0.98), 2),
            "avg_tenure_months": round(rng.uniform(18, 48), 1),
            "job_hop_score": round(rng.uniform(0.05, 0.4), 2),
            "engagement_score": round(rng.uniform(0.4, 0.95), 2),
            "profile_completeness": round(rng.uniform(0.6, 0.98), 2),
        },
    }


def _make_other_candidate(cid: str, rng: random.Random) -> dict:
    skills = rng.sample(OTHER_SKILLS, k=rng.randint(4, 8))
    title = rng.choice(TITLES_OTHER)
    years = rng.randint(1, 8)
    return {
        "candidate_id": cid,
        "profile": {
            "name": f"Candidate {cid}",
            "summary": f"{title} focused on business applications and web development.",
            "location": rng.choice(["Austin", "Chicago", "Boston"]),
            "current_title": title,
        },
        "career_history": [
            {
                "title": title,
                "company": "WebApps Inc",
                "start_date": f"{2024 - years}-01",
                "end_date": "present",
                "description": "Built APIs and internal tools.",
            }
        ],
        "education": [
            {"degree": "BS", "field": "Information Systems", "school": rng.choice(SCHOOLS), "year": 2018}
        ],
        "skills": skills,
        "certifications": [],
        "languages": ["English"],
        "redrob_signals": {
            "profile_views": rng.randint(10, 200),
            "response_rate": round(rng.uniform(0.3, 0.8), 2),
            "avg_tenure_months": round(rng.uniform(12, 36), 1),
            "job_hop_score": round(rng.uniform(0.1, 0.6), 2),
            "engagement_score": round(rng.uniform(0.3, 0.7), 2),
            "profile_completeness": round(rng.uniform(0.4, 0.8), 2),
        },
    }


def _make_honeypot(cid: str) -> dict:
    return {
        "candidate_id": cid,
        "profile": {
            "name": "Too Good To Be True",
            "summary": " ".join(ML_SKILLS * 5),
            "location": "Remote",
            "current_title": "CEO Founder ML Architect",
        },
        "career_history": [],
        "education": [{"degree": "High School", "field": "General", "school": "Local", "year": 2021}],
        "skills": ML_SKILLS * 5,
        "certifications": ["All Certs"],
        "languages": ["English"],
        "redrob_signals": {
            "profile_views": 9999,
            "response_rate": 1.0,
            "avg_tenure_months": 999,
            "job_hop_score": 0.0,
            "engagement_score": 1.0,
            "profile_completeness": 0.05,
        },
    }


def generate_dataset(count: int, output: Path, seed: int = 42) -> None:
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as handle:
        for i in range(count):
            cid = f"cand_{i:06d}"
            roll = rng.random()
            if roll < 0.02:
                record = _make_honeypot(cid)
            elif roll < 0.35:
                record = _make_ml_candidate(cid, rng)
            else:
                record = _make_other_candidate(cid, rng)
            handle.write(json.dumps(record) + "\n")

    print(f"Generated {count} candidates -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic candidates.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "candidates.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.count, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
