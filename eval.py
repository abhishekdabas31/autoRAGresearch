"""
eval.py — FIXED (never edited by agent)

Two evaluation modes:

  fast_mode=True  (default in loop.py --fast)
    - Retrieval only, no LLM call
    - NDCG@10 + Recall@10 using SciFact qrel IDs
    - Runtime: ~10 seconds
    - Comparable to published BeIR/SciFact baselines:
        BM25=0.665, SBERT(all-MiniLM)=0.574, DPR=0.318

  fast_mode=False (default for python eval.py)
    - Full pipeline including generation
    - Composite = 0.50 * Precision@K + 0.30 * answer_relevance + 0.20 * NDCG@10
    - Faithfulness (token-overlap) is REMOVED — it rewards short outputs, not RAG quality
    - Runtime: ~2-3 minutes
"""

import importlib
import math
import sys
import time

from corpus_prep import load_documents, load_eval_set

# Weights for FULL mode composite (faithfulness removed)
METRIC_WEIGHTS_FULL = {
    "precision_at_k": 0.50,   # ID-based: fraction of retrieved docs that are relevant
    "answer_relevance": 0.30,  # token F1 vs gold answer
    "ndcg_at_10": 0.20,        # standard IR metric
}

TIMEOUT_SECONDS = 300


# ------------------------------------------------------------------
# IR metric helpers
# ------------------------------------------------------------------

def _ndcg_at_k(retrieved_ids: list, relevant_ids: set, k: int = 10) -> float:
    """Compute NDCG@k. Assumes binary relevance (relevant=1, not=0)."""
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(retrieved_ids: list, relevant_ids: set, k: int = 10) -> float:
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def _precision_at_k(retrieved_ids: list, relevant_ids: set) -> float:
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
    return hits / len(retrieved_ids)


def _tokenize(text: str) -> set:
    return {w.strip(".,;:!?\"'()[]{}") for w in text.lower().split()} - {""}


def _token_f1(pred: set, gold: set) -> float:
    if not pred or not gold:
        return 0.0
    common = pred & gold
    prec = len(common) / len(pred)
    rec = len(common) / len(gold)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


# ------------------------------------------------------------------
# Main evaluation entry point
# ------------------------------------------------------------------

def run_evaluation(timeout_seconds: int = TIMEOUT_SECONDS, fast_mode: bool = False) -> dict:
    """Run evaluation. Returns metrics dict with composite_score.

    fast_mode=True:  retrieval only, NDCG@10 + Recall@10, ~10s.
    fast_mode=False: full pipeline with generation, ~2-3 min.

    Reloads rag_pipeline on every call so it picks up agent edits.
    """
    if "rag_pipeline" in sys.modules:
        importlib.reload(sys.modules["rag_pipeline"])
    import rag_pipeline

    eval_set = load_eval_set()
    documents = load_documents()
    start_time = time.time()

    try:
        chunks = rag_pipeline.chunk_documents(documents)
        index = rag_pipeline.build_index(chunks)

        ndcg_scores, recall_scores, prec_scores = [], [], []
        answer_rel_scores, latencies = [], []

        for item in eval_set:
            if time.time() - start_time > timeout_seconds:
                return _timeout_result(len(eval_set))

            t0 = time.time()

            relevant_ids = set(item.get("relevant_doc_ids", []))

            if fast_mode:
                # Retrieval only — skip generation entirely
                chunks_retrieved = rag_pipeline.retrieve(item["query"], index)
                # Also apply rerank if enabled (uses no LLM)
                chunks_retrieved = rag_pipeline.rerank(item["query"], chunks_retrieved)
                retrieved_ids = [c["source"] for c in chunks_retrieved]
            else:
                result = rag_pipeline.run_query(item["query"], index)
                retrieved_ids = result.get("source_ids", [c["source"] for c in
                                           rag_pipeline.retrieve(item["query"], index)])
                ans_tok = _tokenize(result["answer"])
                gt_tok = _tokenize(item.get("ground_truth_answer", ""))
                answer_rel_scores.append(_token_f1(ans_tok, gt_tok))

            latencies.append((time.time() - t0) * 1000)
            ndcg_scores.append(_ndcg_at_k(retrieved_ids, relevant_ids, k=10))
            recall_scores.append(_recall_at_k(retrieved_ids, relevant_ids, k=10))
            prec_scores.append(_precision_at_k(retrieved_ids, relevant_ids))

        ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        prec = sum(prec_scores) / len(prec_scores) if prec_scores else 0.0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

        if fast_mode:
            # NDCG@10 is the primary optimization target in fast mode
            composite = ndcg
            return {
                "composite_score": round(composite, 4),
                "ndcg_at_10": round(ndcg, 4),
                "recall_at_10": round(recall, 4),
                "precision_at_k": round(prec, 4),
                "avg_latency_ms": round(avg_lat, 1),
                "n_queries": len(eval_set),
                "mode": "fast",
                "timed_out": False,
            }
        else:
            ans_rel = sum(answer_rel_scores) / len(answer_rel_scores) if answer_rel_scores else 0.0
            composite = (
                METRIC_WEIGHTS_FULL["precision_at_k"] * prec
                + METRIC_WEIGHTS_FULL["answer_relevance"] * ans_rel
                + METRIC_WEIGHTS_FULL["ndcg_at_10"] * ndcg
            )
            return {
                "composite_score": round(composite, 4),
                "ndcg_at_10": round(ndcg, 4),
                "recall_at_10": round(recall, 4),
                "precision_at_k": round(prec, 4),
                "answer_relevance": round(ans_rel, 4),
                "avg_latency_ms": round(avg_lat, 1),
                "n_queries": len(eval_set),
                "mode": "full",
                "timed_out": False,
            }

    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback; traceback.print_exc()
        return _timeout_result(len(eval_set), error=str(e))


def _timeout_result(n_queries: int, error=None) -> dict:
    result = {
        "composite_score": 0.0,
        "ndcg_at_10": 0.0,
        "recall_at_10": 0.0,
        "precision_at_k": 0.0,
        "avg_latency_ms": 0.0,
        "n_queries": n_queries,
        "timed_out": True,
    }
    if error:
        result["error"] = error
    return result


# ------------------------------------------------------------------
# CLI: python eval.py [--fast]
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Retrieval-only eval (NDCG@10, ~10s)")
    args = parser.parse_args()

    mode_label = "FAST (retrieval-only)" if args.fast else "FULL (retrieval + generation)"
    print(f"Running evaluation [{mode_label}]...")
    result = run_evaluation(fast_mode=args.fast)

    if result["timed_out"]:
        print(f"Evaluation timed out or errored: {result.get('error', '')}")
    else:
        print(f"\nComposite Score : {result['composite_score']:.4f}")
        print(f"  NDCG@10       : {result['ndcg_at_10']:.4f}  (published SBERT baseline: 0.574)")
        print(f"  Recall@10     : {result['recall_at_10']:.4f}")
        print(f"  Precision@K   : {result['precision_at_k']:.4f}")
        if not args.fast:
            print(f"  answer_rel    : {result.get('answer_relevance', 0):.4f}")
        print(f"  avg_latency   : {result['avg_latency_ms']:.0f} ms")
        print(f"  queries       : {result['n_queries']}")
        print(f"  mode          : {result['mode']}")
