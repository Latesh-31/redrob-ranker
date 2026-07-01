#!/usr/bin/env bash
set -euo pipefail

# Docker entrypoint for offline usage.
# Expects local model files under /models (mounted as a volume).

MODELS_DIR=${MODELS_DIR:-/models}

echo "Starting entrypoint. MODELS_DIR=${MODELS_DIR}"

if [ ! -d "${MODELS_DIR}" ]; then
  echo "ERROR: ${MODELS_DIR} not found. Mount your local models directory to /models." >&2
  exit 1
fi

# Ensure embedding model exists
if [ ! -f "${MODELS_DIR}/bge-small-en-v1.5/model.safetensors" ] && [ ! -d "${MODELS_DIR}/bge-small-en-v1.5-fp16" ]; then
  echo "ERROR: embedding model not found in ${MODELS_DIR}. Place model files under ${MODELS_DIR}/bge-small-en-v1.5 or bge-small-en-v1.5-fp16" >&2
  exit 1
fi

# Ensure HNSW index exists
if [ ! -f "${MODELS_DIR}/embeddings/candidates_hnsw.bin" ]; then
  echo "Warning: HNSW index not found at ${MODELS_DIR}/embeddings/candidates_hnsw.bin. If missing, rank may be slower." >&2
fi

exec "$@"
