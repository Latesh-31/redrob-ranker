"""Offline preprocessing CLI: load, clean, features, embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.build_features import build_feature_rows, save_features
from src.preprocessing.generate_embeddings import file_sha256, is_local_model_ready
from src.utils.constants import HONEYPOT_HARD_REJECT
from src.preprocessing.load_candidates import load_candidates
from src.utils.constants import DATA_DIR, LOCAL_MODEL_DIR, EMBEDDING_MODEL
from src.utils.logger import get_logger, timed_step

logger = get_logger(__name__)


def download_model_if_needed() -> None:
    if is_local_model_ready():
        return
    logger.info("Downloading embedding model to %s", LOCAL_MODEL_DIR)
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    from src.preprocessing.generate_embeddings import load_embedding_model

    model = load_embedding_model(EMBEDDING_MODEL)
    model.save(str(LOCAL_MODEL_DIR))


def run_preprocess(input_path: Path, skip_embeddings: bool = False, batch_size: int = 64) -> None:
    download_model_if_needed()

    candidates = list(load_candidates(input_path))
    if not candidates:
        raise ValueError(f"No candidates loaded from {input_path}")

    logger.info("Loaded %d candidates", len(candidates))

    with timed_step(logger, "build structured features"):
        rows = build_feature_rows(candidates)
        save_features(rows)

    if skip_embeddings:
        logger.info("Skipping embedding generation")
        return

    # Generate embeddings only for candidates that are not hard honeypot rejects.
    # Keep the full structured features (rows) intact for auditing.
    records_all = [(row["candidate_id"], row["embedding_text"]) for row in rows]
    records = [r for r, row in zip(records_all, rows) if float(row.get("honeypot_penalty", 0.0)) < HONEYPOT_HARD_REJECT]

    with timed_step(logger, "generate embeddings"):
        from src.preprocessing.generate_embeddings import (
            generate_embeddings_from_records,
            load_embedding_model,
            save_embeddings,
        )

        model = load_embedding_model()
        ids, embeddings = generate_embeddings_from_records(records, model=model, batch_size=batch_size)
        source_hash = file_sha256(input_path) if input_path.exists() else ""
        save_embeddings(ids, embeddings, source_hash=source_hash)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess candidate dataset")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "candidates.jsonl")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    run_preprocess(args.input, skip_embeddings=args.skip_embeddings, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
