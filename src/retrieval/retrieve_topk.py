"""Retrieve top-k candidates via embedding similarity."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.retrieval.similarity_search import similarity_search
from src.utils.constants import EMBEDDINGS_DIR, RETRIEVAL_TOP_K, FEATURES_DIR, HONEYPOT_HARD_REJECT
from src.utils.helpers import read_json
import pandas as pd
from pathlib import Path


def load_embedding_artifacts(
    embeddings_dir: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    embeddings_dir = embeddings_dir or EMBEDDINGS_DIR
    embeddings = np.load(embeddings_dir / "candidate_embeddings.npy")
    candidate_ids = read_json(embeddings_dir / "candidate_ids.json")
    # Apply global honeypot filtering using structured features (if available).
    feats_path = Path(FEATURES_DIR) / "structured_features.parquet"
    if feats_path.exists():
        try:
            df = pd.read_parquet(feats_path)
            allowed = set(df.loc[df["honeypot_penalty"] < HONEYPOT_HARD_REJECT, "candidate_id"].astype(str).tolist())
            # Preserve original ordering
            mask = [str(cid) in allowed for cid in candidate_ids]
            if not all(mask):
                filtered_ids = [str(cid) for cid, m in zip(candidate_ids, mask) if m]
                filtered_embeddings = embeddings[np.array(mask, dtype=bool)]
                return filtered_embeddings.astype(np.float32), filtered_ids
        except Exception:
            # If feature loading fails, fall back to returning all embeddings
            pass

    return embeddings.astype(np.float32), [str(cid) for cid in candidate_ids]


def retrieve_topk(
    jd_vector: np.ndarray,
    top_k: int = RETRIEVAL_TOP_K,
    embeddings: np.ndarray | None = None,
    candidate_ids: list[str] | None = None,
    embeddings_dir: Path | None = None,
) -> list[dict[str, float | str]]:
    if embeddings is None or candidate_ids is None:
        embeddings, candidate_ids = load_embedding_artifacts(embeddings_dir)

    indices, scores = similarity_search(jd_vector, embeddings, top_k=top_k)
    return [
        {"candidate_id": candidate_ids[int(idx)], "semantic_score": float(scores[i])}
        for i, idx in enumerate(indices)
    ]
