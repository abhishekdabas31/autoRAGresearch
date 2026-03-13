# Pipeline Theory — Running Research Log

This file is updated by the agent after every experiment.
It is a theory of HOW this pipeline works and fails, not a log of scores.
Scores are in results/experiments.jsonl. This file explains the WHY.

---

## Established Principles (confirmed by multiple experiments)

- Retrieval granularity and context granularity are separate problems.
  Parent-child chunking solves both simultaneously: small chunks for precision
  in matching, full parent for richness in generation. (Exp 10, +0.017 NDCG)

- Rank-based fusion (RRF) is more robust than score-based fusion (linear alpha).
  Linear blending failed because BM25 score variance causes normalization artifacts.
  RRF avoids this by operating on ranks. (Exp 11 kept vs Exp 5 reverted)

- Better embedding models on MTEB transfer to SciFact retrieval gains.
  BGE-small outperformed all-MiniLM by 0.048 NDCG@10. Domain-specific models
  (SPECTER2, SciNCL) likely transfer further. (Exp 9)

- Sentence-window chunking improves recall when using all-MiniLM, because
  sentence-level semantic search is well within all-MiniLM's training distribution.
  (Exp 8, +0.028 NDCG)

## Known Conflicts (technique combinations that hurt)

- sentence_window + parent_child conflict. Both solve retrieval granularity.
  Stacking them creates de-duplication that eliminates candidate documents.
  Do not combine. (Exp 12 reverted, -0.044)

- BGE-small + sentence-level indexing hurts. BGE is trained for passage-level
  retrieval (query vs paragraph). Indexing individual sentences puts the corpus
  out of BGE's training distribution. Use BGE only with whole_doc or parent_child.
  (Exp 9a, regression vs sentence_window baseline)

- BM25 hybrid with linear alpha=0.3 hurts recall. Recall dropped from 0.57 to 0.37.
  Cause: dense signal suppressed by BM25 score normalization artifacts. (Exp 5 reverted)

## Open Failure Modes (not yet addressed)

- Vocabulary gap between short scientific claims and abstract-length documents.
  Hypothesis: HyDE with a capable generator (LLaMA 3.2) should help.
  Not yet tested — flan-t5-small is too weak to generate meaningful hypothetical passages.

- Exact entity matching: scientific queries contain gene names, drug names, proteins.
  Dense models compress these into semantic neighborhoods, losing exact-match signal.
  BM25 captures these but hurts on semantic queries. Adaptive routing or SPLADE
  could solve this without the alpha-blending brittleness.

- BM25 alone scores 0.665 on SciFact vs our current best of 0.5678. This 0.097 gap
  suggests dense retrieval is systematically missing exact-terminology queries. The
  current RRF hybrid helps but hasn't closed the gap — the dense component may be
  diluting BM25's signal on high-specificity queries.

## Unknown / Needs Investigation

- Whether the current NDCG gains are from better embeddings (BGE-small) or from
  parent-child architecture or from RRF. These were stacked in sequence without
  ablation. Isolating contributions would clarify which component to improve next.

- Whether domain-specific embeddings (SPECTER2, SciNCL) add value ON TOP OF the
  current parent-child + RRF architecture, or whether the architecture already
  compensates for general embedding limitations.

- The 0.665 BM25 baseline implies that for SciFact, exact keyword retrieval is
  the dominant signal. Current dense architecture at 0.5678 is 0.087 below BM25.
  A pure BM25-only pipeline as intermediate experiment would be informative.
