# autoRAGresearch 🔬

> Autonomous RAG pipeline optimizer. The agent iterates. You sleep.
> Inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch).

## Current Best Score: **NDCG@10 = 0.5678** (Experiment #11)

> Metric switched to **NDCG@10** (fast retrieval-only, ~90s) in experiment 8. Prior experiments used a full-pipeline composite that included biased faithfulness scores.
> Published SBERT (all-MiniLM-L6-v2) baseline on BeIR/SciFact: **0.574**

| # | NDCG@10 | Delta | Change | Status |
|---|---|---|---|---|
| 11 | **0.5678** | +0.0064 | RRF hybrid retrieval (BM25 + dense) | ✅ Kept |
| 10 | 0.5614 | +0.0167 | parent_child chunking (80-char children, full parent as context) | ✅ Kept |
| 9  | 0.5447 | +0.0483 | BGE-small-en-v1.5 embedding + whole_doc | ✅ Kept |
| 8  | 0.5243 | +0.0279 | sentence_window chunking (WINDOW_SIZE=2) + eval dedup fix | ✅ Kept |
| —  | 0.4964 | baseline | Strategic rebuild: whole_doc + all-MiniLM-L6-v2 + reranker | — |
| 12 | 0.5240 | -0.0438 | sentence_window + BGE-small + RRF (BGE needs passage-length text) | ❌ Reverted |
| 9a | 0.5114 | — | BGE-small + sentence_window (regression, wrong domain for BGE) | ❌ Reverted |

**Total improvement from honest baseline: +0.0714 (+14.4%)** in 5 experiments, closing to within **0.006 of published SBERT baseline**.

---

## The Idea

autoRAGresearch gives an AI agent a real RAG pipeline and lets it experiment autonomously. The agent reads the current `rag_pipeline.py`, forms a hypothesis for improvement, modifies it, evaluates against a fixed query set, commits if improved, reverts if not, and loops.

The human never touches the Python. They only write `program.md` — a Markdown file that gives research direction.

---

## Dataset: BeIR/SciFact

- **Corpus:** 5,183 scientific paper abstracts (from the BEIR benchmark)
- **Eval set:** 30 fixed queries with passage-level relevance labels (qrels)
- **Why SciFact:** Small enough to re-embed per experiment (~20s), clean gold labels, scientific text rewards good chunking and retrieval

```python
# Downloaded automatically by corpus_prep.setup_scifact()
# No manual setup needed
```

---

## Experiment Log

### Experiment 0 — Baseline
**Config:** CHUNK_SIZE=512, CHUNK_OVERLAP=50, top_k=5, all-MiniLM-L6-v2, flan-t5-small
| Metric | Score |
|---|---|
| composite | 0.1236 |
| faithfulness | 0.0333 |
| answer_relevance | 0.0011 |
| context_precision | 0.2328 |
| context_recall | 0.6500 |
| avg_latency | 595 ms |

**Notes:** Large chunks (512 chars) frequently span multiple SciFact abstracts. The embedding model's 256-token limit means tail of long chunks is ignored. flan-t5-small generates verbose answers that barely overlap with ground truth.

---

### Experiment 1 — Chunk size reduction ✅ +0.0325
**Hypothesis:** SciFact abstracts are 150–250 words. CHUNK_SIZE=512 spans multiple abstracts, contaminating chunks. Reducing to 250 chars with zero overlap should align better with abstract boundaries and fit within all-MiniLM-L6-v2's 256-token context window.

**Change:** `CHUNK_SIZE: 512 → 250`, `CHUNK_OVERLAP: 50 → 0`

| Metric | Before | After | Δ |
|---|---|---|---|
| composite | 0.1236 | **0.1561** | +0.0325 |
| faithfulness | 0.0333 | 0.1202 | +0.0869 |
| answer_relevance | 0.0011 | 0.0363 | +0.0352 |
| context_precision | 0.2328 | 0.2067 | -0.0261 |
| context_recall | 0.6500 | 0.6000 | -0.0500 |

**Analysis:** Smaller chunks improved the generation metrics significantly. context_precision/recall dropped slightly — with smaller chunks, the retrieved ids are sub-abstract fragments which don't always match the full passage IDs in qrels.

---

### Experiment 2 — Paragraph chunking ❌ -0.0226
**Hypothesis:** Paragraph splits (`\n\n`) will respect natural abstract boundaries instead of cutting at arbitrary character counts.

**Change:** `CHUNK_STRATEGY: "fixed" → "paragraph"`

**Score:** 0.1561 → 0.1335 (−0.0226). **Reverted.**

**Analysis:** SciFact passages are single-paragraph abstracts with no `\n\n` breaks. Paragraph chunking produced no splits at all for most documents, generating chunks that exceeded the embedding model's 512-token limit (sequence length 631 > 512 warnings). Faithfulness and relevance both dropped.

---

### Experiment 3 — Concise prompt + token limit ✅ +0.3051
**Hypothesis:** flan-t5-small has a 512-token input limit. The current prompt (system instructions + context + question) leaves little room for the context. More critically: without `max_new_tokens` constraint, the model generates very long outputs that diverge from the gold answers. A shorter, directive prompt and `MAX_NEW_TOKENS=64` should improve faithfulness and relevance dramatically.

**Change:** `SYSTEM_PROMPT` simplified, `MAX_NEW_TOKENS: 256 → 64`

| Metric | Before | After | Δ |
|---|---|---|---|
| composite | 0.1561 | **0.4612** | +0.3051 |
| faithfulness | 0.1202 | 0.8932 | +0.7730 |
| answer_relevance | 0.0363 | 0.1349 | +0.0986 |
| context_precision | 0.2067 | 0.2067 | 0.0000 |
| context_recall | 0.6000 | 0.6000 | 0.0000 |

**Analysis:** The single biggest improvement of the run. Faithfulness went from 0.12 → 0.89. Shorter outputs are more grounded in context by construction. This is the most important insight for small T5 models: constrain the output length and the model becomes vastly more faithful.

---

### Experiment 4 — Reduce top_k ✅ +0.0147
**Hypothesis:** With faithfulness now stable, context_precision (0.21) is the next bottleneck. Retrieving fewer chunks means less noise in the context window and higher precision.

**Change:** `RETRIEVAL_TOP_K: 5 → 3`

| Metric | Before | After | Δ |
|---|---|---|---|
| composite | 0.4612 | **0.4759** | +0.0147 |
| faithfulness | 0.8932 | 0.8727 | -0.0205 |
| answer_relevance | 0.1349 | 0.1538 | +0.0189 |
| context_precision | 0.2067 | 0.3000 | +0.0933 |
| context_recall | 0.6000 | 0.5667 | -0.0333 |

**Analysis:** context_precision improved +0.09. Expected recall trade-off accepted. Net positive.

---

### Experiment 5 — Hybrid retrieval ❌ -0.0681
**Hypothesis:** SciFact queries contain specific scientific terms where BM25 lexical matching should outperform dense embeddings. Hybrid retrieval with alpha=0.3 (70% BM25) should recover missed term-specific passages.

**Change:** `USE_HYBRID_RETRIEVAL: False → True`, `HYBRID_ALPHA: 0.5 → 0.3`

**Score:** 0.4759 → 0.4078 (−0.0681). **Reverted.**

**Analysis:** BM25 at this weighting hurt badly — context_recall dropped from 0.57 → 0.37. The normalized score combination may be penalizing the dense signal too heavily at alpha=0.3. Dense retrieval is already capturing the scientific terms well through semantic similarity.

---

### Experiment 6 — T5-native prompt format ❌ -0.0047
**Hypothesis:** flan-t5 was fine-tuned on tasks formatted as `question: {q} context: {c}`. Using this native format instead of the instructional prompt template should improve generation quality.

**Change:** Prompt format changed to `"question: {query} context: {context}"`

**Score:** 0.4759 → 0.4712 (−0.0047). **Reverted.**

**Analysis:** Marginal regression. The instructional format with explicit `Answer:` cue works better than the QA-style format for this setup.

---

### Experiment 7 — Reranker with broader initial retrieval ✅ +0.0022
**Hypothesis:** Retrieve top-10 candidates first for broad recall, then use a cross-encoder (`ms-marco-MiniLM-L-6-v2`) to rerank and select the top-3 highest-quality passages. Should improve both precision and recall over pure top-3 retrieval.

**Change:** `RETRIEVAL_TOP_K: 3 → 10`, `USE_RERANKER: False → True`, `RERANKER_TOP_K: 3`

| Metric | Before | After | Δ |
|---|---|---|---|
| composite | 0.4759 | **0.4781** | +0.0022 |
| faithfulness | 0.8727 | 0.8610 | -0.0117 |
| answer_relevance | 0.1538 | 0.1337 | -0.0201 |
| context_precision | 0.3000 | 0.3500 | +0.0500 |
| context_recall | 0.5667 | 0.6000 | +0.0333 |

**Analysis:** Small net gain. The cross-encoder improved precision and recall, but the slightly noisier context (reranker picks different chunks than dense-only top-3) hurt faithfulness and relevance marginally. Worth keeping for the retrieval quality improvement.

---

## Current Best Configuration

```python
# rag_pipeline.py — current best (score: 0.4781)

CHUNK_STRATEGY = "fixed"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 0

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

RETRIEVAL_TOP_K = 10
USE_HYBRID_RETRIEVAL = False
USE_RERANKER = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 3

LLM_MODEL = "google/flan-t5-small"
MAX_NEW_TOKENS = 64
SYSTEM_PROMPT = "Answer the question using only the given context. Use one or two sentences."
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/abhishekdabas31/autoRAGresearch.git
cd autoRAGresearch
pip install -r requirements.txt

# 2. Download SciFact dataset (one-time, ~30s)
python -c "from corpus_prep import setup_scifact; setup_scifact()"

# 3. Run the current best pipeline
export KMP_DUPLICATE_LIB_OK=TRUE  # required on macOS
python eval.py

# 4. Edit program.md to set your research direction

# 5. Run the autonomous loop (requires ANTHROPIC_API_KEY in .env)
python loop.py --experiments 10
```

---

## How It Works

```
LOOP (runs indefinitely until Ctrl+C):
  1. READ    → load rag_pipeline.py + program.md + recent history
  2. PROMPT  → ask Claude: "propose ONE targeted modification to improve the score"
  3. WRITE   → save agent's proposed rag_pipeline.py (validate syntax first)
  4. EVAL    → run eval.py with timeout
  5. JUDGE   → compare new score vs best — commit if improved, revert if not
  6. LOG     → append experiment record to results/experiments.jsonl
  7. UPDATE  → rewrite README.md leaderboard
  8. GOTO 1
```

## The Three Files That Matter

| File | Who edits it | What it is |
|---|---|---|
| `rag_pipeline.py` | The AI agent | The research canvas |
| `program.md` | The human | The research direction |
| `eval.py` + `corpus_prep.py` | Nobody (fixed) | The arena |

## Repository Structure

```
autoRAGresearch/
├── corpus_prep.py       # FIXED — SciFact loader, embedding + LLM factories
├── rag_pipeline.py      # EDITABLE — agent's canvas, full RAG logic
├── program.md           # HUMAN-WRITTEN — research direction and constraints
├── loop.py              # ORCHESTRATOR — autonomous improvement loop
├── eval.py              # FIXED — evaluation harness, RAGAS-inspired metrics
├── setup_data.py        # Legacy custom corpus generator (superseded by SciFact)
├── results/
│   ├── experiments.jsonl  # append-only log of every experiment
│   └── best_config.json   # snapshot of best-performing config
├── data/
│   ├── corpus.jsonl       # SciFact passages (generated by setup_scifact())
│   ├── corpus/            # legacy custom articles
│   └── eval_set.json      # 30 fixed queries — NEVER CHANGES
├── README.md
└── requirements.txt
```

## What Makes This Different From Optuna

- **Optuna** searches a predefined parameter space you specify in advance.
- **autoRAGresearch** lets the agent *define the search space dynamically* — it can invent new chunking strategies, rewrite retrieval logic, change prompt structure, add techniques from its training knowledge.
- The agent can propose changes no parameter grid would have included.
- The git history is human-readable — you see *why* each change worked.

## Key Findings

### What Works on SciFact
1. **Parent-child chunking** — Indexing sub-phrase children (80 chars) while returning full abstracts as context gives the best of both worlds: precise embedding targets + rich generation context.
2. **BGE-small-en-v1.5 + whole/parent-level indexing** — BGE is trained for passage-level retrieval. Pairing it with passage-length indexed texts (+query instruction prefix) beats all-MiniLM by 4.8 NDCG points.
3. **RRF Hybrid (BM25 + dense)** — SciFact's exact scientific terminology (gene names, drug compounds) favors BM25. RRF fusion stacks on top of dense with no new hyperparameter.
4. **sentence_window + all-MiniLM** — A strong pairing when you want recall gains without changing the embedding model.

### What Doesn't Work
- BGE-small + sentence-level indexing: model is trained on passage-length text; short sentences hurt its representation quality.
- All-MiniLM + BGE's query prefix: the prefix is BGE-specific and likely confuses other models.

## Next Experiments (planned)

- Try `allenai/specter2` — scientific-domain pre-trained, should outperform general BGE on SciFact
- Try `malteos/scincl` — trained specifically on scientific citation graphs
- Increase `CHILD_CHUNK_SIZE` to 150–200 chars (currently 80) to test optimal granularity
- Add title-prepend to child chunks: `{title}: {child_text}` for better topic grounding
- Try `RERANKER_MODEL = "cross-encoder/ms-marco-electra-base"` (stronger reranker)

## CLI Options

```bash
python loop.py                          # run indefinitely
python loop.py --experiments 10         # run exactly 10 experiments
python loop.py --model claude-opus-4-6  # use a specific agent model
python loop.py --dry-run                # propose changes without committing
python loop.py --sleep 10              # pause 10s between experiments
```

## Dependencies

The RAG pipeline runs **entirely on local models** — no API keys needed to run experiments. Only `loop.py` (the orchestration agent) requires an Anthropic key.

## License

MIT
