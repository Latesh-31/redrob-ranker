"""Main entry point for the candidate ranking pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.export.export_csv import export_csv
from src.jd.extract_requirements import extract_requirements
from src.jd.parse_jd import parse_job_description
from src.preprocessing.generate_embeddings import load_embedding_model
from src.reranking.xgb_reranker import RerankResult, XGBReranker
from src.reasoning.generate_reasoning import generate_reasoning
from src.retrieval.embed_jd import embed_job_description
from src.retrieval.retrieve_topk import load_embedding_artifacts, retrieve_topk
from src.scoring.final_score import compute_breakdown
from src.utils.constants import DATA_DIR, FEATURES_DIR, FINAL_TOP_K, HONEYPOT_HARD_REJECT, RETRIEVAL_TOP_K
from src.utils.logger import get_logger, timed_step
from src.utils.types import CandidateFeatures, RankedCandidate

logger = get_logger(__name__)


def load_features_table(features_dir: Path | None = None) -> pd.DataFrame:
    features_dir = features_dir or FEATURES_DIR
    path = features_dir / "structured_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Structured features not found: {path}. Run preprocessing first.")
    df = pd.read_parquet(path)
    df = df.set_index("candidate_id", drop=False)
    return df


def rank_candidates(
    jd_path: Path | None = None,
    retrieval_k: int = RETRIEVAL_TOP_K,
    final_k: int = FINAL_TOP_K,
    output_path: Path | None = None,
    use_xgb_reranker: bool = False,
) -> list[RankedCandidate]:
    start = time.perf_counter()
    jd_path = jd_path or DATA_DIR / "job_description.md"

    with timed_step(logger, "parse job description"):
        parsed_jd = parse_job_description(jd_path)
        requirements = extract_requirements(parsed_jd)

    with timed_step(logger, "load artifacts"):
        model = load_embedding_model()
        embeddings, candidate_ids = load_embedding_artifacts()
        features_df = load_features_table()

    with timed_step(logger, "embed JD and retrieve shortlist"):
        jd_vector = embed_job_description(requirements.embedding_text, model=model)
        shortlist = retrieve_topk(jd_vector, top_k=retrieval_k, embeddings=embeddings, candidate_ids=candidate_ids)
        semantic_map = {item["candidate_id"]: float(item["semantic_score"]) for item in shortlist}

    shortlist_ids = [item["candidate_id"] for item in shortlist]
    shortlist_df = features_df[features_df["candidate_id"].isin(shortlist_ids)]

    shortlist_scored: list[tuple[CandidateFeatures, object, float]] = []
    honeypot_rejects = 0

    with timed_step(logger, f"rerank {len(shortlist_df)} candidates"):
        for row in shortlist_df.to_dict(orient="records"):
            candidate = CandidateFeatures.from_row(row)
            if candidate.honeypot_penalty >= HONEYPOT_HARD_REJECT:
                honeypot_rejects += 1
                continue
            semantic = semantic_map.get(candidate.candidate_id, 0.0)
            breakdown = compute_breakdown(candidate, requirements, semantic_score=semantic)
            shortlist_scored.append((candidate, breakdown, semantic))

    scored: list[tuple[CandidateFeatures, object, float]] = shortlist_scored

    if len(shortlist_scored) < final_k:
        logger.warning(
            "Shortlist yielded %d candidates after filtering; scoring full corpus fallback",
            len(shortlist_scored),
        )
        already = {candidate.candidate_id for candidate, _, _ in shortlist_scored}
        fallback_scored: list[tuple[CandidateFeatures, object, float]] = list(shortlist_scored)
        for row in features_df.to_dict(orient="records"):
            cid = row["candidate_id"]
            if cid in already:
                continue
            candidate = CandidateFeatures.from_row(row)
            if candidate.honeypot_penalty >= HONEYPOT_HARD_REJECT:
                continue
            semantic = semantic_map.get(cid, 0.0)
            breakdown = compute_breakdown(candidate, requirements, semantic_score=semantic)
            fallback_scored.append((candidate, breakdown, semantic))
        scored = fallback_scored

    if use_xgb_reranker and len(shortlist_scored) >= final_k:
        reranker = XGBReranker()
        reranker.fit(shortlist_scored)
        ranked_pool = reranker.rerank(shortlist_scored, minimum_keep=final_k)
    else:
        scored.sort(key=lambda item: (-item[1].final_score, item[0].candidate_id))
        ranked_pool = [
            RerankResult(
                candidate=candidate,
                breakdown=breakdown,
                semantic_score=semantic,
                fit_probability=breakdown.final_score,
                rerank_score=breakdown.final_score,
            )
            for candidate, breakdown, semantic in scored
        ]
    top = ranked_pool[:final_k]

    if len(top) < final_k:
        raise RuntimeError(
            f"Only {len(top)} eligible candidates available; need {final_k}. "
            "Generate more candidates or relax honeypot rules."
        )

    ranked: list[RankedCandidate] = []
    for idx, item in enumerate(top, start=1):
        reasoning = generate_reasoning(item.breakdown, item.candidate, rank=idx)
        ranked.append(
            RankedCandidate(
                candidate_id=item.candidate.candidate_id,
                rank=idx,
                score=item.rerank_score,
                reasoning=reasoning,
                breakdown=item.breakdown,
            )
        )

    export_csv(ranked, path=output_path)

    elapsed = time.perf_counter() - start
    logger.info(
        "Ranking complete: %d results, %d honeypot rejects, %.2fs elapsed",
        len(ranked),
        honeypot_rejects,
        elapsed,
    )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates against job description")
    parser.add_argument("--jd", type=Path, default=DATA_DIR / "job_description.md")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--retrieval-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    parser.add_argument("--use-xgb-reranker", action="store_true")
    args = parser.parse_args()

    rank_candidates(
        jd_path=args.jd,
        retrieval_k=args.retrieval_k,
        final_k=args.top_k,
        output_path=args.output,
        use_xgb_reranker=args.use_xgb_reranker,
    )


if __name__ == "__main__":
    main()
