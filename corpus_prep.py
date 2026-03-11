"""
corpus_prep.py — FIXED (never edited by agent)

Loads documents from the corpus, provides the static eval set,
and exposes factory functions for embedding models and LLM clients.

Supports two corpus formats:
  - SciFact (BeIR benchmark): data/corpus.jsonl + data/eval_set.json with relevant_doc_ids
  - Custom corpus: text files in data/corpus/ + data/eval_set.json
"""

import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CORPUS_JSONL = DATA_DIR / "corpus.jsonl"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_SET_PATH = DATA_DIR / "eval_set.json"

_embedding_cache: dict = {}
_corpus_cache = None  # Optional[list]


def setup_scifact(data_dir=None):
    """One-time download of BeIR/SciFact. Creates corpus.jsonl and eval_set.json."""
    import csv
    import gzip
    import io

    from huggingface_hub import hf_hub_download

    target = Path(data_dir) if data_dir else DATA_DIR
    corpus_path = target / "corpus.jsonl"
    eval_path = target / "eval_set.json"

    if corpus_path.exists() and eval_path.exists():
        print("SciFact data already exists. Skipping download.")
        return

    target.mkdir(parents=True, exist_ok=True)
    print("Downloading SciFact from BeIR benchmark...")

    corpus_gz = hf_hub_download("BeIR/scifact", "corpus.jsonl.gz", repo_type="dataset")
    queries_gz = hf_hub_download("BeIR/scifact", "queries.jsonl.gz", repo_type="dataset")
    qrels_tsv = hf_hub_download("BeIR/scifact-qrels", "train.tsv", repo_type="dataset")

    corpus_lookup = {}
    with gzip.open(corpus_gz, "rt", encoding="utf-8") as gz, \
         open(corpus_path, "w", encoding="utf-8") as out:
        for line in gz:
            doc = json.loads(line)
            entry = {"id": doc["_id"], "text": doc["text"], "title": doc.get("title", "")}
            corpus_lookup[str(doc["_id"])] = entry
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    query_map = {}
    with gzip.open(queries_gz, "rt", encoding="utf-8") as gz:
        for line in gz:
            q = json.loads(line)
            query_map[str(q["_id"])] = q["text"]

    qrel_map = {}
    with open(qrels_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            cid = str(row["corpus-id"])
            qrel_map.setdefault(qid, []).append(cid)

    relevant_qids = list(dict.fromkeys(qrel_map.keys()))[:30]

    eval_set = []
    for qid in relevant_qids:
        if qid not in query_map:
            continue
        doc_ids = qrel_map.get(qid, [])
        gt_contexts = [
            corpus_lookup[did]["text"]
            for did in doc_ids
            if did in corpus_lookup
        ]
        eval_set.append({
            "query": query_map[qid],
            "relevant_doc_ids": doc_ids,
            "ground_truth_answer": " ".join(gt_contexts),
            "ground_truth_contexts": gt_contexts,
        })

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)

    print(f"SciFact ready: {len(corpus_lookup)} passages, {len(eval_set)} eval queries.")


def load_documents() -> list[dict]:
    """Load corpus documents. Prefers corpus.jsonl (SciFact), falls back to corpus/ dir."""
    global _corpus_cache
    if _corpus_cache is not None:
        return _corpus_cache

    if CORPUS_JSONL.exists():
        documents = []
        with open(CORPUS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                title = doc.get("title", "")
                text = f"{title}\n\n{doc['text']}" if title else doc["text"]
                documents.append({
                    "text": text,
                    "source": str(doc["id"]),
                    "metadata": {"id": doc["id"], "title": title},
                })
        if documents:
            _corpus_cache = documents
            return documents

    if not CORPUS_DIR.exists():
        raise FileNotFoundError(
            "No corpus found. Run:\n"
            "  python -c \"from corpus_prep import setup_scifact; setup_scifact()\""
        )

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
            "No documents found. Run:\n"
            "  python -c \"from corpus_prep import setup_scifact; setup_scifact()\""
        )
    _corpus_cache = documents
    return documents


def load_eval_set() -> list[dict]:
    """Load static evaluation set.

    Returns list of dicts with at minimum {query, ground_truth_answer, ground_truth_contexts}.
    SciFact entries also include {relevant_doc_ids} for ID-based retrieval metrics.
    """
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(
            f"Eval set not found: {EVAL_SET_PATH}\n"
            "Run: python -c \"from corpus_prep import setup_scifact; setup_scifact()\""
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
