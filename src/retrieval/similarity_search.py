"""Cosine similarity search over candidate embeddings.

Uses an HNSW index (`candidates_hnsw.bin`) in the embeddings directory if present
for fast approximate nearest-neighbor queries. Falls back to brute-force cosine
search when the index or `hnswlib` is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.utils.constants import EMBEDDINGS_DIR, EMBEDDING_DIM
import json
from pathlib import Path


# Lazy-loaded HNSW index cache
_hnsw_index = None


def _load_hnsw_index(index_path: Path):
    global _hnsw_index
    if _hnsw_index is not None:
        return _hnsw_index

    try:
        import hnswlib
    except Exception:
        return None

    if not index_path.exists():
        return None

    index = hnswlib.Index(space='cosine', dim=EMBEDDING_DIM)
    index.load_index(str(index_path))
    index.set_ef(50)
    _hnsw_index = index
    return _hnsw_index


def similarity_search(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k indices and cosine similarity scores (descending).

    If an HNSW index exists at `EMBEDDINGS_DIR / 'candidates_hnsw.bin'`, use it
    for fast ANN retrieval. Otherwise compute exact cosine scores.
    """
    if candidate_embeddings.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # Try to use an HNSW index if available and its metadata matches the
    # provided in-memory candidate embeddings (so global filtering is respected).
    index_path = Path(EMBEDDINGS_DIR) / 'candidates_hnsw.bin'
    index_meta = Path(str(index_path) + '.meta.json')
    index = _load_hnsw_index(index_path)
    if index is not None and index_meta.exists():
        try:
            meta = json.loads(index_meta.read_text())
            num_indexed = int(meta.get('num_elements', -1))
            if candidate_embeddings is not None and num_indexed == int(candidate_embeddings.shape[0]):
                # Use index only when it was built against the same candidate set
                labels, distances = index.knn_query(query_embedding.astype(np.float32), k=top_k)
                labels = np.array(labels[0], dtype=np.int64)
                distances = np.array(distances[0], dtype=np.float32)
                sims = 1.0 - distances
                return labels, sims
        except Exception:
            # If meta parsing fails, fall back to brute-force
            pass

    # Fallback: brute-force cosine (assuming candidate embeddings are normalized)
    k = min(top_k, len(candidate_embeddings))
    scores = candidate_embeddings @ query_embedding.astype(np.float32)

    if k == len(scores):
        indices = np.argsort(-scores)
        return indices, scores[indices]

    partition = np.argpartition(-scores, k - 1)[:k]
    order = partition[np.argsort(-scores[partition])]
    return order, scores[order]
