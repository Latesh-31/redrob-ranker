# Redrob Ranker

CPU-only two-stage candidate ranking: offline preprocessing (embeddings + structured features) and fast offline ranking (retrieve → rerank → explain → export).

## Architecture

```
Preprocess (offline)          Rank (offline, <5 min)
─────────────────────         ────────────────────────
candidates.jsonl              job_description.md
    → clean / extract             → parse JD + requirements
    → embeddings (.npy)           → embed JD (once)
    → features (.parquet)         → cosine Top-500 retrieval
                                  → XGBoost shortlist filter
                                  → structured rerank Top-100
                                  → template reasoning
                                  → submission.csv
```

## Setup

```bash
cd redrob-ranker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate data (if needed)

```bash
# Synthetic dataset for development (100k)
python scripts/generate_synthetic_data.py --count 100000

# Or place organizer-provided candidates.jsonl in data/
```

## Preprocess (run once)

Downloads BGE-small-en-v1.5 on first run and writes artifacts under `models/`.

```bash
python -m src.preprocessing.preprocess --input data/candidates.jsonl
```

Artifacts:
- `models/embeddings/candidate_embeddings.npy`
- `models/embeddings/candidate_ids.json`
- `models/features/structured_features.parquet`
- `models/features/skill_vocab.json`

## Rank

```bash
python rank.py --jd data/job_description.md --output output/submission.csv
python validate_submission.py
```

The rerank stage now includes an optional XGBoost fit classifier on the semantic shortlist. Use `--use-xgb-reranker` to enable it; otherwise the CLI uses the stable heuristic rerank path.

## Tests

```bash
pytest tests/ -q
```

## Scoring fusion

| Signal     | Weight |
|------------|--------|
| Semantic   | 0.20   |
| Skills     | 0.30   |
| Experience | 0.25   |
| Title      | 0.15   |
| Education  | 0.05   |
| Behavior   | 0.05   |

Honeypot penalty multiplicatively reduces final score; hard rejects (penalty ≥ 0.8) are excluded.

## Performance notes

- Ranking loads precomputed embeddings — no candidate encoding at rank time.
- Target ranking latency: ~10–15s for 100k candidates on CPU.
- Preprocessing 100k embeddings: ~30–90 min CPU (one-time).

## Offline usage

This project supports fully offline operation. Put your model files and indexes under the local `models/` directory before running.

Required local artifacts (place under `models/`):
- `bge-small-en-v1.5/` or `bge-small-en-v1.5-fp16/` — the embedding model files (local copy of `model.safetensors` or FP16 directory).
- `embeddings/candidate_embeddings.npy` and `embeddings/candidate_ids.json` — candidate embeddings.
- `embeddings/candidates_hnsw.bin` — optional HNSW index for fast retrieval (recommended).

Run locally (activate venv first):

```bash
source .venv/bin/activate
python rank.py --jd data/job_description.md --output output/submission.csv
```

Or use the convenience script:

```bash
./scripts/run_offline.sh
```

If you prefer to run inside Docker, mount your local `models/` into the container at `/models`:

```bash
# Build image (no model files baked in)
docker build -t redrob-ranker:offline .

# Run with host models mounted (no network required)
docker run --rm -v $(pwd)/models:/models -v $(pwd)/output:/app/output redrob-ranker:offline
```

### Build a self-contained offline image

If you want a single image that contains the model files (so the container can run without any mounts), build the offline image locally. Note: the build step downloads Python packages once and therefore requires network access during the build; the resulting image runs offline.

```bash
# Ensure models/ contains your model and index
./scripts/build_offline_image.sh redrob-ranker:offline-packed

# Run the packed image (no mounts required)
docker run --rm -v $(pwd)/output:/app/output redrob-ranker:offline-packed
```


