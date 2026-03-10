"""
eval.py — FIXED (never edited by agent)

Evaluation harness. Imports rag_pipeline, runs every query from the eval set,
computes metrics, and returns a single comparable composite score.

Metric weights (configurable):
  faithfulness      0.35  — Is the answer grounded in retrieved context?
  answer_relevance  0.35  — Does the answer address the query correctly?
  context_precision 0.20  — Are the retrieved chunks actually relevant?
  context_recall    0.10  — Were the relevant chunks actually retrieved?
"""

import importlib
import sys
import time

from corpus_prep import load_documents, load_eval_set

METRIC_WEIGHTS = {
    "faithfulness": 0.35,
    "answer_relevance": 0.35,
    "context_precision": 0.20,
    "context_recall": 0.10,
}

TIMEOUT_SECONDS = 300


# ------------------------------------------------------------------
# Token-level metric helpers (no LLM evaluator needed)
# ------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lowercase whitespace tokenization with basic punctuation stripping."""
    return {w.strip(".,;:!?\"'()[]{}") for w in text.lower().split()} - {""}


def _token_f1(pred: set[str], gold: set[str]) -> float:
    if not pred or not gold:
        return 0.0
    common = pred & gold
    prec = len(common) / len(pred)
    rec = len(common) / len(gold)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def _compute_metrics(results: list[dict]) -> dict:
    """Compute RAGAS-inspired metrics using token overlap heuristics.

    These are fast, deterministic, and require no LLM evaluator.
    For LLM-judge evaluation, integrate the ragas library with Ollama.
    """
    faith, relevance, ctx_prec, ctx_rec = [], [], [], []

    for r in results:
        ans_tok = _tokenize(r["answer"])
        ctx_tok = _tokenize(" ".join(r["contexts"]))
        gt_ans_tok = _tokenize(r["ground_truth_answer"])
        gt_ctx_tok = _tokenize(" ".join(r["ground_truth_contexts"]))

        # Faithfulness: fraction of answer tokens grounded in context
        faith.append(len(ans_tok & ctx_tok) / len(ans_tok) if ans_tok else 0.0)

        # Answer relevance: F1 between predicted and gold answer
        relevance.append(_token_f1(ans_tok, gt_ans_tok))

        # Context precision: avg relevance of each retrieved chunk
        if r["contexts"]:
            chunk_scores = []
            for ctx in r["contexts"]:
                ct = _tokenize(ctx)
                chunk_scores.append(len(ct & gt_ctx_tok) / len(ct) if ct else 0.0)
            ctx_prec.append(sum(chunk_scores) / len(chunk_scores))
        else:
            ctx_prec.append(0.0)

        # Context recall: coverage of ground-truth context by retrieved chunks
        if gt_ctx_tok:
            all_ret = _tokenize(" ".join(r["contexts"]))
            ctx_rec.append(len(gt_ctx_tok & all_ret) / len(gt_ctx_tok))
        else:
            ctx_rec.append(0.0)

    return {
        "faithfulness": sum(faith) / len(faith) if faith else 0.0,
        "answer_relevance": sum(relevance) / len(relevance) if relevance else 0.0,
        "context_precision": sum(ctx_prec) / len(ctx_prec) if ctx_prec else 0.0,
        "context_recall": sum(ctx_rec) / len(ctx_rec) if ctx_rec else 0.0,
    }


# ------------------------------------------------------------------
# Main evaluation entry point
# ------------------------------------------------------------------

def run_evaluation(timeout_seconds: int = TIMEOUT_SECONDS) -> dict:
    """Run the full eval pipeline. Returns metrics dict with composite_score.

    Reloads rag_pipeline on every call so it picks up agent edits.
    Aborts with score 0 if the eval exceeds timeout_seconds.
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

        results, latencies = [], []
        for item in eval_set:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                return _timeout_result(len(eval_set))

            t0 = time.time()
            result = rag_pipeline.run_query(item["query"], index)
            latencies.append((time.time() - t0) * 1000)

            results.append({
                "query": item["query"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth_answer": item["ground_truth_answer"],
                "ground_truth_contexts": item["ground_truth_contexts"],
            })

        metrics = _compute_metrics(results)
        composite = sum(metrics[k] * METRIC_WEIGHTS[k] for k in METRIC_WEIGHTS)

        return {
            "composite_score": round(composite, 4),
            "faithfulness": round(metrics["faithfulness"], 4),
            "answer_relevance": round(metrics["answer_relevance"], 4),
            "context_precision": round(metrics["context_precision"], 4),
            "context_recall": round(metrics["context_recall"], 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "n_queries": len(eval_set),
            "timed_out": False,
        }

    except Exception as e:
        print(f"Evaluation error: {e}")
        return _timeout_result(len(eval_set), error=str(e))


def _timeout_result(n_queries: int, error: str | None = None) -> dict:
    result = {
        "composite_score": 0.0,
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "avg_latency_ms": 0.0,
        "n_queries": n_queries,
        "timed_out": True,
    }
    if error:
        result["error"] = error
    return result


# ------------------------------------------------------------------
# CLI: python eval.py
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Running evaluation...")
    result = run_evaluation()
    if result["timed_out"]:
        print(f"⚠  Evaluation timed out or errored. {result.get('error', '')}")
    else:
        print(f"\nComposite Score: {result['composite_score']:.4f}")
        print(f"  faithfulness:      {result['faithfulness']:.4f}  (weight {METRIC_WEIGHTS['faithfulness']})")
        print(f"  answer_relevance:  {result['answer_relevance']:.4f}  (weight {METRIC_WEIGHTS['answer_relevance']})")
        print(f"  context_precision: {result['context_precision']:.4f}  (weight {METRIC_WEIGHTS['context_precision']})")
        print(f"  context_recall:    {result['context_recall']:.4f}  (weight {METRIC_WEIGHTS['context_recall']})")
        print(f"  avg_latency:       {result['avg_latency_ms']:.0f} ms")
        print(f"  queries:           {result['n_queries']}")
