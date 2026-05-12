"""
memory/store.py
---------------
Qdrant vector store wrapper — runs fully locally (no server needed).
Handles storing, retrieving, and managing personal memory chunks.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter, FieldCondition,
    MatchValue, ScoredPoint,
    PayloadSchemaType,
)

from ingestion.chunker import Chunk


@dataclass
class MemoryResult:
    """A retrieved memory with its relevance score."""
    chunk_id: str
    text: str
    score: float                   # cosine similarity (0–1, higher = more relevant)
    source: str
    source_type: str
    title: str
    chunk_index: int
    total_chunks: int
    created_at: datetime
    metadata: dict
    is_stale: bool = False         # True if memory is older than staleness threshold
    trust_weight: float = 1.0      # adjusted score after staleness penalty


class MemoryStore:
    """
    Local Qdrant-based vector store for personal memories.
    Stores chunks as points with full metadata in the payload.
    """

    def __init__(
        self,
        db_path: str = "./data/qdrant_db",
        collection_name: str = "personal_memory",
        embedding_dim: int = 384,        # all-MiniLM-L6-v2 dimension
        staleness_days: int = 365,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.staleness_days = staleness_days

        # Create DB directory
        Path(db_path).mkdir(parents=True, exist_ok=True)

        # Connect to local Qdrant
        self.client = QdrantClient(path=db_path)

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Create the collection if it doesn't exist yet."""
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            print(f"  ✓ Created memory collection: '{self.collection_name}'")
        else:
            count = self.client.count(self.collection_name).count
            print(f"  ✓ Connected to memory store: '{self.collection_name}' ({count} memories)")

    def add_chunks(
        self,
        chunk_embedding_pairs: list[tuple[Chunk, list[float]]],
        skip_duplicates: bool = True,
    ) -> int:
        """
        Store a list of (Chunk, embedding) pairs in the vector store.
        Returns the number of chunks actually stored.
        """
        if not chunk_embedding_pairs:
            return 0

        # Get existing chunk IDs to skip duplicates
        existing_ids = set()
        if skip_duplicates:
            existing_ids = self._get_existing_chunk_ids(
                [c.chunk_id for c, _ in chunk_embedding_pairs]
            )

        points = []
        skipped = 0

        for chunk, embedding in chunk_embedding_pairs:
            if chunk.chunk_id in existing_ids:
                skipped += 1
                continue

            # Store created_at as ISO string (Qdrant doesn't have native datetime)
            payload = {
                "chunk_id":    chunk.chunk_id,
                "text":        chunk.text,
                "source":      chunk.source,
                "source_type": chunk.source_type,
                "title":       chunk.title,
                "chunk_index": chunk.chunk_index,
                "total_chunks":chunk.total_chunks,
                "created_at":  chunk.created_at.isoformat(),
                "word_count":  chunk.word_count,
                **{f"meta_{k}": str(v) for k, v in chunk.metadata.items()},
            }

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload,
            ))

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

        stored = len(points)
        if skipped:
            print(f"  ↩ Skipped {skipped} duplicate chunks")
        if stored:
            print(f"  ✓ Stored {stored} new memory chunks")

        return stored

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 0.35,
        source_filter: Optional[str] = None,
    ) -> list[MemoryResult]:
        """
        Semantic search over personal memories.
        Returns top_k results above the score threshold, with staleness scoring.
        """
        search_filter = None
        if source_filter:
            search_filter = Filter(
                must=[FieldCondition(
                    key="source",
                    match=MatchValue(value=source_filter)
                )]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k * 2,             # fetch extra, filter below threshold
            score_threshold=score_threshold,
            with_payload=True,
        )


        memories = []
        now = datetime.now(timezone.utc)

        for hit in results:
            p = hit.payload
            created_at = datetime.fromisoformat(p["created_at"])

            # Make timezone-aware for comparison
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            age_days = (now - created_at).days
            is_stale = age_days > self.staleness_days

            # Staleness penalty: linear decay after threshold
            if is_stale:
                staleness_factor = max(0.5, 1.0 - (age_days - self.staleness_days) / 365)
            else:
                staleness_factor = 1.0

            trust_weight = hit.score * staleness_factor

            memories.append(MemoryResult(
                chunk_id=p["chunk_id"],
                text=p["text"],
                score=round(hit.score, 4),
                source=p["source"],
                source_type=p["source_type"],
                title=p["title"],
                chunk_index=p["chunk_index"],
                total_chunks=p["total_chunks"],
                created_at=created_at,
                metadata={k: v for k, v in p.items() if k.startswith("meta_")},
                is_stale=is_stale,
                trust_weight=round(trust_weight, 4),
            ))

        # Sort by trust_weight (staleness-adjusted score)
        memories.sort(key=lambda m: m.trust_weight, reverse=True)
        return memories[:top_k]

    def count(self) -> int:
        """Return total number of stored memory chunks."""
        return self.client.count(self.collection_name).count

    def list_sources(self) -> list[dict]:
        """Return all unique sources stored in memory."""
        # Scroll all points and collect unique sources
        sources = {}
        offset = None

        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=["source", "source_type", "title", "created_at"],
            )

            for point in results:
                p = point.payload
                src = p["source"]
                if src not in sources:
                    sources[src] = {
                        "source": src,
                        "source_type": p["source_type"],
                        "title": p["title"],
                        "created_at": p["created_at"],
                        "chunks": 0,
                    }
                sources[src]["chunks"] += 1

            if offset is None:
                break

        return list(sources.values())

    def delete_source(self, source: str) -> int:
        """Delete all chunks from a specific source."""
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )
        print(f"  🗑 Deleted memories from: {source}")
        return result.status

    def _get_existing_chunk_ids(self, chunk_ids: list[str]) -> set[str]:
        """Check which chunk IDs already exist (for deduplication)."""
        existing = set()
        # Scroll and check - simple approach for small stores
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=["chunk_id"],
            )
            for point in results:
                cid = point.payload.get("chunk_id")
                if cid in chunk_ids:
                    existing.add(cid)
            if offset is None:
                break
        return existing
