# rag_pipeline.py — AGENT-EDITABLE
# This is the only file the agent modifies. Every parameter is intentionally surfaced.
# The agent reads this file, hypothesizes improvements, and rewrites it.

import numpy as np
from corpus_prep import load_documents, get_embedding_model

# ============================================================
# SECTION 1: CONFIGURATION — All tunable parameters live here
# ============================================================

# Chunking
CHUNK_STRATEGY = "fixed"  # options: "fixed", "sentence", "paragraph", "semantic", "recursive"
CHUNK_SIZE = 250  # tokens/chars depending on strategy
CHUNK_OVERLAP = 0  # overlap between consecutive chunks

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# Retrieval
RETRIEVAL_TOP_K = 5
USE_HYBRID_RETRIEVAL = False  # combine dense + BM25 sparse retrieval
HYBRID_ALPHA = 0.5  # weight for dense vs sparse (0=pure sparse, 1=pure dense)
USE_RERANKER = False  # apply cross-encoder reranking post-retrieval
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 3

# Generation
LLM_MODEL = "google/flan-t5-small"
GENERATION_TEMPERATURE = 0.1
MAX_NEW_TOKENS = 64
SYSTEM_PROMPT = "Answer the question using only the given context. Use one or two sentences."

_llm_pipeline = None

# Query Processing
USE_QUERY_EXPANSION = False
QUERY_EXPANSION_N = 3
USE_HyDE = False  # Hypothetical Document Embedding

# Context Assembly
CONTEXT_SEPARATOR = "\n\n---\n\n"
MAX_CONTEXT_LENGTH = 2048


# ============================================================
# SECTION 2: PIPELINE LOGIC — Agent can modify these functions
# ============================================================


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents according to CHUNK_STRATEGY and CHUNK_SIZE."""
    chunks = []
    for doc in documents:
        text = doc["text"]
        source = doc["source"]

        if CHUNK_STRATEGY == "fixed":
            step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
            for i in range(0, len(text), step):
                chunk_text = text[i : i + CHUNK_SIZE]
                if chunk_text.strip():
                    chunks.append({"text": chunk_text, "source": source, "start": i})

        elif CHUNK_STRATEGY == "sentence":
            import re

            sentences = re.split(r"(?<=[.!?])\s+", text)
            current, start = "", 0
            for sent in sentences:
                if len(current) + len(sent) > CHUNK_SIZE and current.strip():
                    chunks.append({"text": current.strip(), "source": source, "start": start})
                    start += len(current)
                    current = sent + " "
                else:
                    current += sent + " "
            if current.strip():
                chunks.append({"text": current.strip(), "source": source, "start": start})

        elif CHUNK_STRATEGY == "paragraph":
            offset = 0
            for para in text.split("\n\n"):
                if para.strip():
                    chunks.append({"text": para.strip(), "source": source, "start": offset})
                offset += len(para) + 2

        else:
            step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
            for i in range(0, len(text), step):
                chunk_text = text[i : i + CHUNK_SIZE]
                if chunk_text.strip():
                    chunks.append({"text": chunk_text, "source": source, "start": i})

    return chunks


def build_index(chunks: list[dict]) -> dict:
    """Build retrieval index (FAISS for dense, optionally BM25 for hybrid)."""
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


def retrieve(query: str, index: dict, top_k=None) -> list[dict]:
    """Retrieve relevant chunks. Supports dense-only or hybrid (dense + BM25)."""
    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    model = get_embedding_model(EMBEDDING_MODEL)
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")

    scores, indices = index["faiss_index"].search(q_emb, top_k)

    if USE_HYBRID_RETRIEVAL and index["bm25"] is not None:
        bm25_scores = index["bm25"].get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:top_k]

        dense_results = {int(i): float(s) for i, s in zip(indices[0], scores[0]) if i >= 0}
        sparse_results = {int(i): float(bm25_scores[i]) for i in bm25_top}

        max_d = max(dense_results.values(), default=1.0) or 1.0
        max_s = max(sparse_results.values(), default=1.0) or 1.0

        combined = {}
        for idx in set(dense_results) | set(sparse_results):
            d = dense_results.get(idx, 0.0) / max_d
            s = sparse_results.get(idx, 0.0) / max_s
            combined[idx] = HYBRID_ALPHA * d + (1 - HYBRID_ALPHA) * s

        top_indices = sorted(combined, key=combined.get, reverse=True)[:top_k]
    else:
        top_indices = [int(i) for i in indices[0] if i >= 0]

    n = len(index["chunks"])
    return [index["chunks"][i] for i in top_indices if i < n]


def rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Apply cross-encoder reranking if USE_RERANKER is True."""
    if not USE_RERANKER or not chunks:
        return chunks

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(RERANKER_MODEL)
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[: RERANKER_TOP_K]]


def assemble_context(chunks: list[dict]) -> str:
    """Concatenate retrieved chunks into a context string for the LLM."""
    context = CONTEXT_SEPARATOR.join(c["text"] for c in chunks)
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH]
    return context


def generate_answer(query: str, context: str) -> str:
    """Generate answer using a local HuggingFace model (no external API)."""
    global _llm_pipeline
    if _llm_pipeline is None:
        from transformers import pipeline as hf_pipeline
        _llm_pipeline = hf_pipeline(
            "text2text-generation",
            model=LLM_MODEL,
            max_new_tokens=MAX_NEW_TOKENS,
        )
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
