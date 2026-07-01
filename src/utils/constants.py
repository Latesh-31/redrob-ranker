"""Project-wide constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
FEATURES_DIR = MODELS_DIR / "features"
OUTPUT_DIR = ROOT_DIR / "output"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_MODEL_DIR = MODELS_DIR / "bge-small-en-v1.5"
EMBEDDING_DIM = 384

RETRIEVAL_TOP_K = 500
FINAL_TOP_K = 100
EMBED_BATCH_SIZE = 64

HONEYPOT_HARD_REJECT = 0.80
MAX_EMBEDDING_CHARS = 4000

SCORE_WEIGHTS = {
    "semantic": 0.20,
    "skill": 0.30,
    "experience": 0.25,
    "title": 0.15,
    "education": 0.05,
    "behavior": 0.05,
}

REQUIRED_CANDIDATE_KEYS = ("candidate_id",)

SIGNAL_NAMES = (
    "profile_views",
    "response_rate",
    "avg_tenure_months",
    "job_hop_score",
    "engagement_score",
    "profile_completeness",
)

DEGREE_LEVELS = {
    "phd": 4,
    "doctorate": 4,
    "doctoral": 4,
    "mba": 3,
    "ms": 3,
    "m.s": 3,
    "master": 3,
    "masters": 3,
    "bs": 2,
    "b.s": 2,
    "bachelor": 2,
    "bachelors": 2,
    "associate": 1,
    "high school": 0,
    "diploma": 0,
}

SENIORITY_KEYWORDS = {
    "junior": "junior",
    "entry": "junior",
    "mid": "mid",
    "middle": "mid",
    "senior": "senior",
    "lead": "senior",
    "staff": "senior",
    "principal": "senior",
    "architect": "senior",
}

SKILL_ALIASES = {
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "ml": "machine learning",
    "dl": "deep learning",
    "tf": "tensorflow",
    "pt": "pytorch",
    "sentence transformers": "sentence-transformers",
    "vector db": "vector search",
    "faiss": "vector search",
}

HONEYPOT_SKILL_BLOCKLIST = frozenset(
    {
        "python",
        "pytorch",
        "tensorflow",
        "scikit-learn",
        "nlp",
        "embeddings",
        "ranking",
        "sql",
        "aws",
        "kubernetes",
        "docker",
        "spark",
        "machine learning",
        "deep learning",
    }
)
