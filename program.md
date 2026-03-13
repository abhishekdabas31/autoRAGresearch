# autoRAGresearch — Research Program

## Dataset and Eval Mode
- **Corpus:** BeIR/SciFact — 5,183 scientific abstracts
- **Eval:** NDCG@10 (fast mode, ~90s per experiment)
- **Published baselines:** BM25=0.665, SBERT(all-MiniLM)=0.574, DPR=0.318
- **Current best:** Check results/best_config.json

---

## Your Job: Researcher, Not Tuner

You are not optimizing parameters. You are discovering *why* this pipeline fails
on specific queries and implementing architectural fixes for those failure modes.

The difference:
- A tuner says: "I'll try chunk_size=200, maybe that helps."
- A researcher says: "Query 7 retrieves the wrong doc because the question uses
  the word 'mechanism' while the relevant abstract uses 'pathway'. This is a
  vocabulary gap. I'll implement HyDE to bridge it by generating an intermediate
  document."

**Every experiment must start from a specific failure diagnosis.**

---

## MANDATORY PROTOCOL — Do This Before Every Proposal

### Step 1: Diagnose current failures
Look at the per-query scores in the experiment history. Identify 2-3 queries that
are consistently scoring low (NDCG < 0.3 for those queries). Ask: what is the
failure mode? Options are:
- **Vocabulary gap:** query uses different words than the relevant document
- **Granularity mismatch:** right information is in a specific sentence but
  the retrieved chunk is too broad or too narrow
- **Ranking failure:** the right doc is retrieved but ranked too low
- **Multi-aspect query:** query requires multiple facts from different documents
- **Specificity mismatch:** query is very specific but index has it at wrong level

### Step 2: Propose one architectural change that addresses that failure mode
The change must add new code — a new function, a new retrieval strategy, a new
processing step. It cannot be a number change alone.

Ask yourself: "If I removed all the configuration constants and kept only the
functions, would this experiment still represent a different approach?" If yes,
it's valid. If no, it's a parameter change. Do not submit it.

### Step 3: Predict specifically which queries will improve and why
Name the failure mode. Name the queries. Name the mechanism. If you can't do this,
you don't have a hypothesis — you have a guess. Go back to Step 1.

### Step 4: Update the theory
After every experiment (pass or fail), write one sentence explaining what this
result taught you about how the pipeline works. This goes in THEORY_UPDATE.
"The experiment failed" is not a valid theory update. "Parent-child and
sentence-window conflict because both solve retrieval granularity — combining
them creates redundant de-duplication that hurts recall" is a valid theory update.

---

## What Counts as Innovation vs Tuning

### INNOVATION (valid experiments):
- Implementing a new retrieval function (adaptive routing, iterative retrieval,
  late chunking, FLARE-style dynamic retrieval)
- Changing how queries are represented before retrieval (HyDE, step-back prompting,
  multi-query expansion)
- Changing the indexing architecture (multi-granularity index, summary+chunk index)
- Adding a new scoring/reranking step with new logic
- Changing the fundamental flow of the pipeline (retrieve→generate→re-retrieve)

### TUNING (invalid, will be flagged):
- Changing CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K, HYBRID_ALPHA alone
- Swapping a model string without implementing new surrounding logic
- Changing prompt templates or MAX_NEW_TOKENS
- Re-trying a previously failed technique with different constants

Model swaps are only valid when paired with architectural justification:
"I'm switching to SPECTER2 because SPECTER2 encodes citation-graph relationships
which directly addresses the vocabulary gap failure mode in queries 3, 7, 14."
A model swap with no failure-mode reasoning is still tuning.

---

## Response Format — Use This Exactly

```
FAILURE_ANALYSIS: [which 2-3 queries are failing, what failure mode they exhibit,
what evidence in the retrieved results points to that failure mode]

MECHANISM: [how your proposed change directly addresses that failure mode —
be specific about the causal chain, not just "this should improve score"]

THEORY_UPDATE: [one sentence updating the running theory of how this pipeline
works — should add new information, not restate the hypothesis]

CHANGE: [one sentence: what you implemented or swapped]
---
[complete modified rag_pipeline.py]
```

---

## Technique Directions (Not a Menu — Use Only When Failure Analysis Points Here)

**Vocabulary gap failures → HyDE or Step-back prompting**
Generate an intermediate document that bridges query vocabulary to document
vocabulary. HyDE is wired in (`USE_HyDE = True`). Prefer a capable generator
(LLaMA 3.2 via Ollama if available) over flan-t5-small, which generates
3-word fragments rather than hypothetical passages.

**Granularity mismatch → Late chunking or multi-granularity index**
Current parent-child is already implemented. Late chunking is the next level:
embed the whole document, derive chunk embeddings from document embedding via
attention pooling — this preserves document-level context in chunk representations.

**Ranking failure → Learned sparse retrieval (SPLADE) or adaptive routing**
BM25=0.665 beats dense retrieval on SciFact because scientific queries use exact
terminology. SPLADE learns to expand both query and document at index time,
combining BM25's exact-match strength with semantic generalization. This is
architecturally different from the current BM25+dense hybrid.

**Multi-aspect queries → Iterative or multi-hop retrieval**
After initial retrieval, identify what information is still missing from the
context. Generate a follow-up query targeting the gap. Retrieve again. Merge
context. This is a fundamental change to the pipeline flow.

**Specificity mismatch → Query decomposition**
Break complex queries into sub-questions. Retrieve for each sub-question
independently. Merge and deduplicate. Good for queries about relationships
between concepts.

---

## Efficiency Constraint

Latency matters. If a technique adds more than 3x latency for less than 0.02 NDCG
gain, it is not worth keeping. Always report avg_latency_ms alongside NDCG@10.
A technique that gets 0.59 NDCG at 200ms is more valuable than 0.60 NDCG at 4000ms.

---

## Fixed Files (Never Edit)
- `corpus_prep.py` — data loading, do not touch
- `eval.py` — evaluation arena, do not touch
- `loop.py` — orchestrator, do not touch
