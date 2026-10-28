"""
src/indexer.py - Asterline Research Assistant
Creates, maintains, and searches the persistent local Chroma vector index.
Stores all vector data under data/chroma_db/ using local embeddings.
"""

import os
from typing import Any, Dict, List, Optional, Set

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Base project root resolved relative to this source file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_db")
DEFAULT_CORPUS_PATH = os.path.join(PROJECT_ROOT, "corpus.jsonl")
COLLECTION_NAME = "asterline_corpus"


def get_embedding_function() -> embedding_functions.DefaultEmbeddingFunction:
    """
    Returns Chroma's built-in local ONNX embedding function (all-MiniLM-L6-v2).
    Runs 100% locally on CPU without remote API calls or PyTorch dependencies.
    """
    return embedding_functions.DefaultEmbeddingFunction()


def get_chroma_client(db_path: str = DEFAULT_DB_PATH) -> chromadb.PersistentClient:
    """
    Initializes and returns a persistent ChromaDB client pointing to db_path.
    Creates parent directories if they do not exist.
    """
    os.makedirs(db_path, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)


def get_or_create_collection(
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = COLLECTION_NAME,
    db_path: str = DEFAULT_DB_PATH,
) -> chromadb.Collection:
    """
    Retrieves or creates the persistent collection using cosine distance
    and the local embedding function.
    """
    if client is None:
        client = get_chroma_client(db_path)

    ef = get_embedding_function()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def ensure_indexed(
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = COLLECTION_NAME,
    db_path: str = DEFAULT_DB_PATH,
    corpus_path: Optional[str] = None,
) -> int:
    """
    Ensures that the persistent collection contains the indexed corpus records.
    If the collection is empty (e.g. cold-start deployment on Streamlit Cloud
    or a fresh container), loads corpus.jsonl and indexes it idempotently.

    Reuses existing `load_corpus` and `index_corpus` without duplication or recursion.
    Returns the total number of records in the collection (expected: 75).
    """
    collection = get_or_create_collection(
        client=client, collection_name=collection_name, db_path=db_path
    )
    current_count = collection.count()
    if current_count > 0:
        return current_count

    if corpus_path:
        target_corpus = corpus_path
        if not os.path.exists(target_corpus):
            raise FileNotFoundError(
                f"Cold-start indexing failed: corpus file not found at '{target_corpus}'."
            )
    else:
        target_corpus = DEFAULT_CORPUS_PATH
        if not os.path.exists(target_corpus):
            alt_corpus = os.path.join(os.getcwd(), "corpus.jsonl")
            if os.path.exists(alt_corpus):
                target_corpus = alt_corpus
            else:
                raise FileNotFoundError(
                    f"Cold-start indexing failed: corpus file not found at '{target_corpus}'."
                )

    from src.corpus import load_corpus

    try:
        records = load_corpus(target_corpus)
        return index_corpus(
            records=records,
            client=client,
            collection_name=collection_name,
            db_path=db_path,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cold-start auto-indexing failed for '{target_corpus}': {exc}"
        ) from exc


def index_corpus(
    records: List[Dict[str, Any]],
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = COLLECTION_NAME,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    Indexes validated corpus records into the persistent Chroma collection.
    
    IDEMPOTENCY:
    Uses `collection.upsert(...)` with stable `chunk_id` values.
    Running indexing multiple times updates existing records without creating
    duplicates, guaranteeing the total count remains strictly equal to the number
    of unique source chunks (75).
    
    Returns the total number of records currently in the collection.
    """
    collection = get_or_create_collection(
        client=client, collection_name=collection_name, db_path=db_path
    )

    if not records:
        return collection.count()

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for r in records:
        chunk_id = str(r["chunk_id"])
        ids.append(chunk_id)
        documents.append(str(r["text"]))

        # Preserve all original metadata fields for citations and filtering
        meta = {
            "chunk_id": chunk_id,
            "doc_id": str(r["doc_id"]),
            "title": str(r["title"]),
            "category": str(r["category"]),
            "owner": str(r["owner"]),
            "source_url": str(r["source_url"]),
            "published": str(r["published"]),
            "version": str(r["version"]),
        }
        metadatas.append(meta)

    # Upsert ensures idempotent insertion
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    return collection.count()


def get_all_indexed_ids(
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = COLLECTION_NAME,
    db_path: str = DEFAULT_DB_PATH,
) -> Set[str]:
    """
    Returns the set of all chunk IDs currently stored in the index.
    """
    collection = get_or_create_collection(
        client=client, collection_name=collection_name, db_path=db_path
    )
    result = collection.get()
    return set(result["ids"]) if result and "ids" in result else set()


def search_corpus(
    query: str,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = COLLECTION_NAME,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Performs semantic vector search against the persistent corpus index.
    Automatically initializes the index if the collection is empty (cold-start).
    
    Args:
        query: User or research query string.
        top_k: Number of nearest chunks to retrieve.
        where: Optional Chroma metadata filter (e.g., {"category": "pricing"}).
        client: Optional PersistentClient instance.
        collection_name: Target collection name.
        db_path: Path to Chroma persistent storage.
        
    Returns:
        List of dictionaries with full chunk fields, preserved original text,
        retrieval distance, and similarity score.
    """
    collection = get_or_create_collection(
        client=client, collection_name=collection_name, db_path=db_path
    )

    # Cold-start safeguard: auto-index if collection is unpopulated
    if collection.count() == 0:
        ensure_indexed(
            client=client,
            collection_name=collection_name,
            db_path=db_path,
        )

    query_params: Dict[str, Any] = {
        "query_texts": [query],
        "n_results": top_k,
    }
    if where:
        query_params["where"] = where

    raw_results = collection.query(**query_params)

    hits: List[Dict[str, Any]] = []
    if not raw_results or not raw_results.get("ids") or not raw_results["ids"][0]:
        return hits

    ids = raw_results["ids"][0]
    docs = raw_results["documents"][0] if raw_results.get("documents") else []
    metas = raw_results["metadatas"][0] if raw_results.get("metadatas") else []
    distances = raw_results["distances"][0] if raw_results.get("distances") else []

    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        doc_text = docs[i] if i < len(docs) else ""
        dist = distances[i] if i < len(distances) else 0.0

        # For cosine space in Chroma, distance in [0, 2], similarity = 1 - (dist / 2) or 1 - dist
        similarity_score = max(0.0, 1.0 - float(dist))

        hit = {
            "chunk_id": ids[i],
            "doc_id": meta.get("doc_id", ""),
            "title": meta.get("title", ""),
            "category": meta.get("category", ""),
            "owner": meta.get("owner", ""),
            "source_url": meta.get("source_url", ""),
            "published": meta.get("published", ""),
            "version": meta.get("version", ""),
            "text": doc_text,
            "distance": round(float(dist), 4),
            "score": round(similarity_score, 4),
        }
        hits.append(hit)

    return hits
