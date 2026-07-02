#!/usr/bin/env bash
set -euo pipefail

# Run the project offline using the local Python venv and local models.
# Usage: ./scripts/run_offline.sh

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT_DIR/.venv/bin/activate"

export MODELS_DIR=${MODELS_DIR:-$ROOT_DIR/models}

echo "Using MODELS_DIR=$MODELS_DIR"

python rank.py --jd data/job_description.md --output output/submission.xlsx
