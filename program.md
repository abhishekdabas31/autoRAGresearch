# autoRAGresearch — Research Program

## Dataset and Eval Mode
- **Corpus:** BeIR/SciFact — 5,183 scientific abstracts (passage-level retrieval unit)
- **Eval:** NDCG@10 (fast mode, ~10s) — comparable to published BeIR baselines
- **Published baselines for reference:** BM25=0.665 NDCG@10, SBERT(all-MiniLM)=0.574, DPR=0.318
- **Current starting point:** Run `python eval.py --fast` to measure

## Objective
Maximize NDCG@10 on the 30 fixed SciFact eval queries by implementing RAG techniques
that would transfer to any scientific or domain-specific corpus — not just SciFact.

---

## THE ONE RULE FOR ALL EXPERIMENTS

**Every experiment must implement a new function or swap an embedding model.**
A change that can be described as "set X from A to B" is not a valid experiment.
The test: could this change transfer to a completely different corpus with the same code?
If yes, it's valid. If it relies on SciFact-specific tuning, it's not.

---

## Research Roadmap — 3 Tiers, in Priority Order

### Tier 1 — Retrieval Architecture Techniques
These require writing new code. They are the highest-leverage and most generalizable experiments.

---

**Technique A: Sentence-Window Retrieval**

What it is: Index documents at the sentence level for precise matching. When a sentence
is retrieved, return the surrounding ±WINDOW_SIZE sentences as the actual context.
This gives precision of sentence-level indexing with the coherence of paragraph-level context.

Why it matters: SciFact abstracts contain 4-8 sentences. A single relevant sentence is often
more precisely matchable than a full 250-char chunk. But one sentence alone lacks context
for a good answer. Sentence-window solves both problems simultaneously.

How to implement: `CHUNK_STRATEGY = "sentence_window"` is already wired in rag_pipeline.py.
The `chunk_documents()` function handles it when this strategy is set. It stores both the
sentence (for indexing) and `context_window` (for generation). The `assemble_context()`
function already uses `context_window` when this strategy is active. Set `WINDOW_SIZE = 2`.

Expected gain: Should improve NDCG@10 by surfacing more precise sentence-level matches
while keeping generated answers coherent. Generalizes to any corpus with multi-sentence passages.

---

**Technique B: Parent-Child Chunking**

What it is: Index small child chunks (~80 chars) for precise retrieval. Each child stores
its parent passage ID. After retrieval, deduplicate by parent ID and return the full parent
passage as context. Decouples retrieval precision from context quality.

Why it matters: Dense retrieval on 250-char chunks forces a precision-recall tradeoff.
Small chunks = precise matching but thin context. Large chunks = rich context but noisy matching.
Parent-child breaks this tradeoff: retrieval sees small units, generation sees full passages.

How to implement: `CHUNK_STRATEGY = "parent_child"` is wired in rag_pipeline.py.
`chunk_documents()` creates child chunks with `full_parent_text` stored.
`assemble_context()` deduplicates by source and uses `full_parent_text`.
Set `CHILD_CHUNK_SIZE = 80`.

Expected gain: Should increase both Precision@K and Recall@10 simultaneously.
Classic technique from LlamaIndex, transfers to any corpus.

---

**Technique C: HyDE (Hypothetical Document Embeddings)**

What it is: Instead of embedding the raw query, generate a synthetic answer passage first
(using the LLM), then embed that as the retrieval query. The synthetic passage has more
vocabulary overlap with the actual abstracts than the short question does.

Why it matters: The query "Does CRISPR edit DNA?" is short and sparse. The relevant abstract
begins "CRISPR-Cas9 is a genome editing tool that cleaves double-stranded DNA...". The
vocabulary gap between query and document reduces recall. A synthetic answer paragraph
bridges this gap.

How to implement: `USE_HyDE = True` is wired in rag_pipeline.py.
The `generate_hypothetical_doc(query)` function already exists — it calls the local LLM
to produce a hypothetical passage, which is then embedded by `retrieve()`.
Warning: adds one LLM call per query, so fast_mode eval will be slower. Test at ~20-30s.

Expected gain: Known to improve recall by 5-15% on scientific corpora (Gao et al. 2022).
Generalizes to any domain with vocabulary mismatch between queries and documents.

---

**Technique D: Multi-Query Retrieval**

What it is: Generate N rephrasings of the query, retrieve independently for each,
union all result sets before reranking. Improves recall on queries that have multiple
valid phrasings or are ambiguous.

Why it matters: "What enzyme is responsible for DNA repair?" might miss a passage that
answers "Nuclease activity in DNA damage response." A rephrasing like "Which protein
performs DNA repair?" would catch it. Union retrieval finds both.

How to implement: `USE_QUERY_EXPANSION = True` with `QUERY_EXPANSION_N = 3`.
The `expand_query(query, n)` function already exists — it calls the LLM to generate
rephrasings and `retrieve()` unions results across all expanded queries.
Also adds LLM overhead, similar to HyDE.

Expected gain: +5-10% recall, especially on queries with specialized terminology.
Transfers to any corpus with vocabulary-diverse queries.

---

### Tier 2 — Embedding Models
These are model swaps (one-line changes to EMBEDDING_MODEL). Implement after at least one
Tier 1 technique succeeds. Each swap should be combined with the best Tier 1 architecture.

**Models to try in this order:**

1. `BAAI/bge-small-en-v1.5` — MTEB leaderboard top performer, same size as all-MiniLM,
   trained with better contrastive objectives. Drop-in swap. Published NDCG@10 on SciFact: ~0.60.

2. `allenai/specter2` (with `allenai/specter2_base`) — Trained on scientific paper citations.
   Built specifically for scientific literature retrieval. Likely outperforms general-purpose
   models on SciFact. Slightly larger (110M). Use `trust_remote_code=True`.

3. `malteos/scincl` — Trained on biomedical + scientific paper relationships.
   SciNCL uses citation graph data similar to SciFact's domain.

Note: Always run `python eval.py --fast` before and after an embedding model swap to confirm
the change actually improved NDCG@10 and wasn't just a model size change.

---

### Tier 3 — Fusion Techniques
Implement these after Tier 1 and Tier 2 are stable. These are architectural changes to
how dense and sparse signals are combined.

**Reciprocal Rank Fusion (RRF) for hybrid retrieval:**

What it is: When combining BM25 and dense retrieval, use RRF instead of linear score
blending. RRF is parameter-free — it combines rankings rather than scores, which avoids
the normalization brittleness that caused BM25 alpha=0.3 to fail.

How to implement: Set `USE_HYBRID_RETRIEVAL = True` and `FUSION_METHOD = "rrf"`.
The `_rrf_fusion()` function already exists in rag_pipeline.py and `retrieve()` calls it
when `FUSION_METHOD = "rrf"`.

Why previous BM25 failed: We used linear blending with max-norm normalization. BM25 scores
on SciFact have high variance, causing the normalization to suppress the dense signal.
RRF avoids this entirely — it's rank-based, not score-based.

Expected gain: Previously hybrid retrieval lost 0.07 composite score. With RRF, it should
recover at minimum to neutral and potentially add +0.02-0.05 on recall.

---

## Experiments NOT to run

These are explicitly out of scope because they are parameter changes, not technique implementations:

- Changing CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K, HYBRID_ALPHA, MAX_NEW_TOKENS alone
- Changing GENERATION_TEMPERATURE
- Changing RERANKER_TOP_K alone
- Re-trying paragraph or sentence chunking with different sizes
- Any change that only modifies a constant, not a function or model

If you find yourself proposing a number change, stop and propose a technique instead.

---

## How to know if an experiment is valid

Ask: "Could this exact code run unchanged on a financial document corpus or a legal corpus?"
If yes — the experiment is valid. It implements a generalizable technique.
If no — the change is SciFact-specific tuning. Do not run it.
