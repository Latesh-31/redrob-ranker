"""Generate and persist candidate embeddings."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.constants import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBED_BATCH_SIZE,
    EMBEDDINGS_DIR,
    LOCAL_MODEL_DIR,
)
from src.utils.helpers import write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def is_local_model_ready() -> bool:
    return (LOCAL_MODEL_DIR / "config.json").exists() or (LOCAL_MODEL_DIR / "modules.json").exists()


def resolve_model_path() -> str:
    if is_local_model_ready():
        return str(LOCAL_MODEL_DIR)
    return EMBEDDING_MODEL


def load_embedding_model(model_path: str | None = None) -> SentenceTransformer:
    path = model_path or resolve_model_path()
    logger.info("Loading embedding model from %s", path)
    device = "cpu"
    logger.info("Using device: %s", device)
    return SentenceTransformer(path, device=device)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (matrix / norms).astype(np.float32)


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    import torch
    
    chunk_size = 5120
    all_embeddings = []
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        logger.info("Encoding chunk %d/%d (%d texts)", i // chunk_size + 1, (len(texts) + chunk_size - 1) // chunk_size, len(chunk))
        embeddings = model.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        all_embeddings.append(embeddings)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
    final_embeddings = np.vstack(all_embeddings)
    return l2_normalize(final_embeddings)


def save_embeddings(
    candidate_ids: list[str],
    embeddings: np.ndarray,
    output_dir: Path | None = None,
    source_hash: str = "",
) -> tuple[Path, Path, Path]:
    output_dir = output_dir or EMBEDDINGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_path = output_dir / "candidate_embeddings.npy"
    ids_path = output_dir / "candidate_ids.json"
    meta_path = output_dir / "embedding_metadata.json"

    np.save(emb_path, embeddings)
    write_json(ids_path, candidate_ids)
    write_json(
        meta_path,
        {
            "model": resolve_model_path(),
            "embedding_dim": EMBEDDING_DIM,
            "count": len(candidate_ids),
            "source_hash": source_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    logger.info("Saved %d embeddings to %s", len(candidate_ids), emb_path)
    return emb_path, ids_path, meta_path


def generate_embeddings_from_records(
    records: Iterable[tuple[str, str]],
    model: SentenceTransformer | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    texts: list[str] = []
    for cid, text in records:
        ids.append(cid)
        texts.append(text or " ")

    if not ids:
        return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    model = model or load_embedding_model()
    embeddings = encode_texts(model, texts, batch_size=batch_size)
    return ids, embeddings


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
