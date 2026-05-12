"""
ingestion/embedder.py
---------------------
Convert text chunks into dense vector embeddings using
sentence-transformers (runs fully on CPU, no GPU needed).
Model: all-MiniLM-L6-v2 — 384 dimensions, fast, great quality.
"""

import os
import time
from typing import Optional
from ingestion.chunker import Chunk

# Lazy import so startup is fast
_model = None
_model_name = None


def _get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load embedding model once and cache it."""
    global _model, _model_name

    if _model is None or _model_name != model_name:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            )
        print(f"  🔄 Loading embedding model: {model_name}")
        print(f"     (First load downloads ~90MB — subsequent loads are instant)")
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        print(f"  ✓ Model loaded — embedding dimension: {_model.get_sentence_embedding_dimension()}")

    return _model


def embed_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Embed a list of text strings.
    Returns a list of float vectors (one per text).
    """
    model = _get_model(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize for cosine similarity via dot product
    )

    return embeddings.tolist()


def embed_query(
    query: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[float]:
    """
    Embed a single query string for retrieval.
    Always use this (not embed_texts) for queries — keeps model consistent.
    """
    model = _get_model(model_name)
    embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.tolist()


def embed_chunks(
    chunks: list[Chunk],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> list[tuple[Chunk, list[float]]]:
    """
    Embed a list of Chunk objects.
    Returns list of (chunk, embedding) pairs.
    """
    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]

    print(f"\n  🧠 Embedding {len(chunks)} chunks...")
    start = time.time()

    embeddings = embed_texts(texts, model_name=model_name, batch_size=batch_size)

    elapsed = time.time() - start
    print(f"  ✓ Done in {elapsed:.1f}s ({len(chunks)/elapsed:.1f} chunks/sec)")

    return list(zip(chunks, embeddings))


def get_embedding_dimension(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> int:
    """Return the vector dimension for the given model."""
    model = _get_model(model_name)
    return model.get_sentence_embedding_dimension()
