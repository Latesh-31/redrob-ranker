#!/usr/bin/env bash
set -euo pipefail

# Build a self-contained Docker image that includes the local `models/` directory.
# The build step will install Python packages from PyPI (network required for build).

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
MODELS_DIR=${MODELS_DIR:-$ROOT_DIR/models}
IMAGE_TAG=${1:-redrob-ranker:offline}

if [ ! -d "$MODELS_DIR" ]; then
  echo "ERROR: models directory not found at $MODELS_DIR" >&2
  echo "Place your model files (bge-small-en-v1.5 or bge-small-en-v1.5-fp16) and embeddings under $MODELS_DIR" >&2
  exit 1
fi

echo "Building Docker image $IMAGE_TAG (this step requires network access to download pip packages)"
docker build -f Dockerfile.offline -t "$IMAGE_TAG" "$ROOT_DIR"

echo "Built $IMAGE_TAG"
