"""Embed job description text for similarity search."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.preprocessing.generate_embeddings import encode_texts, load_embedding_model


def embed_job_description(jd_text: str, model: SentenceTransformer | None = None) -> np.ndarray:
    model = model or load_embedding_model()
    vector = encode_texts(model, [jd_text or " "])[0]
    return vector.astype(np.float32)
