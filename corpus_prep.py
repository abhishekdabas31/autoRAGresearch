"""
corpus_prep.py — FIXED (never edited by agent)

Loads documents from the corpus directory, provides the static eval set,
and exposes factory functions for embedding models and LLM clients.
"""

import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_SET_PATH = DATA_DIR / "eval_set.json"

_embedding_cache: dict = {}


def load_documents() -> list[dict]:
    """Load raw documents from data/corpus/. Supports .txt, .md, .pdf."""
    if not CORPUS_DIR.exists():
        raise FileNotFoundError(f"Corpus directory not found: {CORPUS_DIR}")

    documents = []
    for path in sorted(CORPUS_DIR.iterdir()):
        if path.suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8")
            if text.strip():
                documents.append({
                    "text": text,
                    "source": path.name,
                    "metadata": {"path": str(path)},
                })
        elif path.suffix == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                if text.strip():
                    documents.append({
                        "text": text,
                        "source": path.name,
                        "metadata": {"path": str(path)},
                    })
            except ImportError:
                print(f"Warning: pypdf not installed, skipping {path.name}")

    if not documents:
        raise ValueError(
            "No documents found. Run `python setup_data.py` to generate the default corpus."
        )
    return documents


def load_eval_set() -> list[dict]:
    """Load static evaluation set.

    Returns list of {query, ground_truth_answer, ground_truth_contexts} dicts.
    """
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(
            f"Eval set not found: {EVAL_SET_PATH}\n"
            "Run `python setup_data.py` to generate it."
        )
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding_model(name: str = "nomic-ai/nomic-embed-text-v1.5"):
    """Return a SentenceTransformer model (cached after first load)."""
    if name not in _embedding_cache:
        from sentence_transformers import SentenceTransformer

        _embedding_cache[name] = SentenceTransformer(name, trust_remote_code=True)
    return _embedding_cache[name]


def get_llm(model_name: str = "llama3.2:3b"):
    """Return an Ollama client and model name tuple for local inference."""
    import ollama

    client = ollama.Client()
    try:
        client.show(model_name)
    except Exception:
        print(
            f"Warning: Model '{model_name}' not found locally. "
            f"Run: ollama pull {model_name}"
        )
    return client, model_name
