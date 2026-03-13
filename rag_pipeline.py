# rag_pipeline.py — AGENT-EDITABLE
# This is the only file the agent modifies.
# Every parameter is intentionally surfaced. Every technique hook is pre-wired.
#
# THE RULE FOR THIS FILE:
#   Valid experiments implement a NEW FUNCTION or swap a MODEL.
#   Changing a constant value alone is NOT a valid experiment.
#   If your change can be described as "set X=Y", it will be rejected.

import numpy as np
from corpus_prep import load_documents, get_embedding_model

# ============================================================
# SECTION 1: CONFIGURATION — All tunable parameters live here
# ============================================================

# Chunking
# "whole_doc" = one document per chunk (correct for SciFact — preserves passage IDs)
# "fixed"     = character-level splitting (use CHUNK_SIZE + CHUNK_OVERLAP)
# "sentence"  = sentence-boundary splitting
# "sentence_window" = index sentences, return surrounding window as context
# "parent_child"    = index small child chunks, return parent passage as context
CHUNK_STRATEGY = "parent_child"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 0
WINDOW_SIZE = 2       # for sentence_window: sentences before/after match to include
CHILD_CHUNK_SIZE = 80 # for parent_child: size of child chunks for indexing

# Embedding
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 32

# Retrieval
RETRIEVAL_TOP_K = 10
USE_HYBRID_RETRIEVAL = False
HYBRID_ALPHA = 0.7    # weight for dense (0=pure sparse, 1=pure dense)
FUSION_METHOD = "linear"  # "linear" or "rrf" (Reciprocal Rank Fusion)
RRF_K = 60            # RRF constant (standard default)

USE_RERANKER = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 5

# Query Processing
USE_QUERY_EXPANSION = False   # multi-query: generate N rephrasings and union results
QUERY_EXPANSION_N = 3
USE_HyDE = False              # generate a hypothetical answer, embed it as the query

# Generation
LLM_MODEL = "google/flan-t5-small"
MAX_NEW_TOKENS = 64
SYSTEM_PROMPT = "Answer the question using only the given context. Use one or two sentences."

_llm_pipeline = None

# Context Assembly
CONTEXT_SEPARATOR = "\n\n---\n\n"
MAX_CONTEXT_LENGTH = 2048


# ============================================================
# SECTION 2: PIPELINE LOGIC — Agent implements techniques here
# ============================================================


def chunk_documents(documents: list) -> list:
    """Chunk documents according to CHUNK_STRATEGY.

    whole_doc      — returns each doc as-is (preserves source IDs for qrel matching)
    fixed          — character-level windows
    sentence       — sentence-boundary groups
    sentence_window — index sentences, context includes surrounding window
    parent_child   — index small children, retrieve returns parent passages
    """
    chunks = []

    if CHUNK_STRATEGY == "whole_doc":
        for doc in documents:
            if doc["text"].strip():
                chunks.append({
                    "text": doc["text"],
                    "source": doc["source"],
                    "start": 0,
                })
        return chunks

    if CHUNK_STRATEGY == "sentence_window":
        import re
        for doc in documents:
            sents = re.split(r"(?<=[.!?])\s+", doc["text"].strip())
            sents = [s for s in sents if s.strip()]
            for i, sent in enumerate(sents):
                window_start = max(0, i - WINDOW_SIZE)
                window_end = min(len(sents), i + WINDOW_SIZE + 1)
                context_window = " ".join(sents[window_start:window_end])
                chunks.append({
                    "text": sent,
                    "source": doc["source"],
                    "start": i,
                    "context_window": context_window,
                })
        return chunks

    if CHUNK_STRATEGY == "parent_child":
        for doc in documents:
            text = doc["text"]
            step = max(CHILD_CHUNK_SIZE, 1)
            for i in range(0, len(text), step):
                child_text = text[i: i + CHILD_CHUNK_SIZE].strip()
                if child_text:
                    chunks.append({
                        "text": child_text,
                        "source": doc["source"],   # parent ID preserved
                        "start": i,
                        "full_parent_text": text,  # full passage for context assembly
                    })
        return chunks

    if CHUNK_STRATEGY == "sentence":
        import re
        for doc in documents:
            sentences = re.split(r"(?<=[.!?])\s+", doc["text"])
            current, start = "", 0
            for sent in sentences:
                if len(current) + len(sent) > CHUNK_SIZE and current.strip():
                    chunks.append({"text": current.strip(), "source": doc["source"], "start": start})
                    start += len(current)
                    current = sent + " "
                else:
                    current += sent + " "
            if current.strip():
                chunks.append({"text": current.strip(), "source": doc["source"], "start": start})
        return chunks

    # Default: fixed-size character windows
    for doc in documents:
        text = doc["text"]
        step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
        for i in range(0, len(text), step):
            chunk_text = text[i: i + CHUNK_SIZE]
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "source": doc["source"], "start": i})

    return chunks


def build_index(chunks: list) -> dict:
    """Build retrieval index (FAISS dense + optional BM25 for hybrid)."""
    import faiss

    model = get_embedding_model(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    index_data = {
        "chunks": chunks,
        "faiss_index": faiss_index,
        "embeddings": embeddings,
        "bm25": None,
    }

    if USE_HYBRID_RETRIEVAL:
        from rank_bm25 import BM25Okapi
        tokenized = [t.lower().split() for t in texts]
        index_data["bm25"] = BM25Okapi(tokenized)

    return index_data


def generate_hypothetical_doc(query: str) -> str:
    """HyDE: generate a hypothetical answer passage to use as the retrieval query.

    The synthetic passage is embedded instead of the raw query, bridging
    the vocabulary gap between short queries and longer scientific passages.
    """
    global _llm_pipeline
    if _llm_pipeline is None:
        from transformers import pipeline as hf_pipeline
        _llm_pipeline = hf_pipeline("text2text-generation", model=LLM_MODEL, max_new_tokens=128)
    prompt = f"Write a short scientific passage that would answer: {query}"
    result = _llm_pipeline(prompt, max_new_tokens=128)
    return result[0]["generated_text"]


def expand_query(query: str, n: int = QUERY_EXPANSION_N) -> list:
    """Multi-query: generate N rephrasings of the query for broader retrieval coverage."""
    global _llm_pipeline
    if _llm_pipeline is None:
        from transformers import pipeline as hf_pipeline
        _llm_pipeline = hf_pipeline("text2text-generation", model=LLM_MODEL, max_new_tokens=64)
    queries = [query]
    for _ in range(n - 1):
        prompt = f"Rephrase this scientific question differently: {query}"
        result = _llm_pipeline(prompt, max_new_tokens=64)
        rephrased = result[0]["generated_text"].strip()
        if rephrased and rephrased != query:
            queries.append(rephrased)
    return list(dict.fromkeys(queries))  # dedup, preserve order


def _rrf_fusion(dense_ranked: list, sparse_ranked: list, k: int = RRF_K) -> list:
    """Reciprocal Rank Fusion — parameter-free, robust alternative to linear blending."""
    scores = {}
    for rank, idx in enumerate(dense_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, idx in enumerate(sparse_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def retrieve(query: str, index: dict, top_k=None) -> list:
    """Retrieve relevant chunks. Supports dense, hybrid, multi-query, and HyDE."""
    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    model = get_embedding_model(EMBEDDING_MODEL)

    # HyDE: embed a synthetic answer passage instead of the raw query
    retrieval_query = generate_hypothetical_doc(query) if USE_HyDE else query

    # Multi-query: union results from multiple query rephrasings
    if USE_QUERY_EXPANSION:
        all_queries = expand_query(retrieval_query)
    else:
        all_queries = [retrieval_query]

    # BGE models require a query instruction prefix for asymmetric retrieval
    bge_prefix = "Represent this sentence for searching relevant passages: "
    use_bge_prefix = "bge" in EMBEDDING_MODEL.lower()

    seen, merged_chunks = set(), []
    for q in all_queries:
        encoded_q = (bge_prefix + q) if use_bge_prefix else q
        q_emb = model.encode([encoded_q], normalize_embeddings=True).astype("float32")
        scores_arr, indices_arr = index["faiss_index"].search(q_emb, top_k)

        if USE_HYBRID_RETRIEVAL and index["bm25"] is not None:
            bm25_scores = index["bm25"].get_scores(q.lower().split())
            bm25_ranked = list(np.argsort(bm25_scores)[::-1][:top_k])
            dense_ranked = [int(i) for i in indices_arr[0] if i >= 0]

            if FUSION_METHOD == "rrf":
                top_indices = _rrf_fusion(dense_ranked, bm25_ranked)[:top_k]
            else:
                # Linear alpha fusion
                max_d = max((float(s) for s in scores_arr[0] if s > 0), default=1.0)
                max_s = max((float(bm25_scores[i]) for i in bm25_ranked), default=1.0)
                combined = {}
                for idx in set(dense_ranked) | set(bm25_ranked):
                    d = (scores_arr[0][dense_ranked.index(idx)] / max_d
                         if idx in dense_ranked else 0.0)
                    s = float(bm25_scores[idx]) / max_s if idx < len(bm25_scores) else 0.0
                    combined[idx] = HYBRID_ALPHA * d + (1 - HYBRID_ALPHA) * s
                top_indices = sorted(combined, key=combined.get, reverse=True)[:top_k]
        else:
            top_indices = [int(i) for i in indices_arr[0] if i >= 0]

        n = len(index["chunks"])
        for i in top_indices:
            if i < n and i not in seen:
                seen.add(i)
                merged_chunks.append(index["chunks"][i])

    return merged_chunks[:top_k]


def rerank(query: str, chunks: list) -> list:
    """Apply cross-encoder reranking if USE_RERANKER is True."""
    if not USE_RERANKER or not chunks:
        return chunks
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(RERANKER_MODEL)
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:RERANKER_TOP_K]]


def assemble_context(chunks: list) -> str:
    """Concatenate retrieved chunks into a context string for the LLM.

    For sentence_window strategy, uses the full context window.
    For parent_child strategy, deduplicates and uses full parent passages.
    """
    if CHUNK_STRATEGY == "sentence_window":
        parts = [c.get("context_window", c["text"]) for c in chunks]
    elif CHUNK_STRATEGY == "parent_child":
        seen_sources = set()
        parts = []
        for c in chunks:
            if c["source"] not in seen_sources:
                seen_sources.add(c["source"])
                parts.append(c.get("full_parent_text", c["text"]))
    else:
        parts = [c["text"] for c in chunks]

    context = CONTEXT_SEPARATOR.join(parts)
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH]
    return context


def generate_answer(query: str, context: str) -> str:
    """Generate answer using a local HuggingFace model."""
    global _llm_pipeline
    if _llm_pipeline is None:
        from transformers import pipeline as hf_pipeline
        _llm_pipeline = hf_pipeline("text2text-generation", model=LLM_MODEL, max_new_tokens=MAX_NEW_TOKENS)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    result = _llm_pipeline(prompt, max_new_tokens=MAX_NEW_TOKENS)
    return result[0]["generated_text"]


def run_query(query: str, index: dict) -> dict:
    """Full pipeline: retrieve → rerank → assemble → generate."""
    chunks = retrieve(query, index)
    chunks = rerank(query, chunks)
    context = assemble_context(chunks)
    answer = generate_answer(query, context)
    return {
        "answer": answer,
        "contexts": [c["text"] for c in chunks],
        "source_ids": [c["source"] for c in chunks],
        "query": query,
    }
