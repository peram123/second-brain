"""
ingestion/chunker.py
--------------------
Split raw documents into smart, overlapping chunks.
Chunks by paragraph boundaries (not fixed token counts) so each
chunk is semantically coherent — a key quality signal for RAG.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from ingestion.loaders import RawDocument


@dataclass
class Chunk:
    """A single memory unit ready for embedding and storage."""
    text: str
    chunk_id: str                  # unique: "{source_slug}_{index}"
    source: str                    # original file path or URL
    source_type: str               # pdf / docx / txt / url / text
    title: str
    chunk_index: int               # position in document
    total_chunks: int              # total chunks from this document
    created_at: datetime = field(default_factory=datetime.now)
    word_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.word_count = len(self.text.split())


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove junk characters."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by double newlines."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _merge_short_paragraphs(paragraphs: list[str], min_words: int = 20) -> list[str]:
    """
    Merge very short paragraphs with the next one.
    Prevents tiny chunks that carry no useful semantic meaning.
    """
    merged = []
    buffer = ""

    for para in paragraphs:
        if buffer:
            combined = buffer + "\n\n" + para
            if len(buffer.split()) < min_words:
                buffer = combined
                continue
            else:
                merged.append(buffer)
                buffer = para
        else:
            buffer = para

    if buffer:
        merged.append(buffer)

    return merged


def _split_long_paragraph(text: str, max_words: int, overlap_words: int) -> list[str]:
    """
    Split a single very long paragraph into overlapping word-window chunks.
    Used only when a paragraph exceeds max_words.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += max_words - overlap_words

    return chunks


def chunk_document(
    doc: RawDocument,
    chunk_size: int = 400,        # target words per chunk
    chunk_overlap: int = 50,      # overlap words between chunks
    min_chunk_words: int = 20,    # discard chunks smaller than this
) -> list[Chunk]:
    """
    Convert a RawDocument into a list of Chunks.

    Strategy:
    1. Clean the text
    2. Split by paragraph boundaries (semantic units)
    3. Merge tiny paragraphs
    4. Group paragraphs into chunks up to chunk_size words
    5. Add overlap by carrying the last paragraph forward
    6. Split any remaining oversized paragraphs
    """
    clean = _clean_text(doc.content)
    if not clean:
        return []

    paragraphs = _split_into_paragraphs(clean)
    paragraphs = _merge_short_paragraphs(paragraphs, min_words=min_chunk_words)

    # Build source slug for chunk IDs
    import hashlib
    source_slug = hashlib.md5(doc.source.encode()).hexdigest()[:8]

    chunks_text = []
    current_parts = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If single paragraph is too long, split it independently
        if para_words > chunk_size:
            # Flush current buffer first
            if current_parts:
                chunks_text.append("\n\n".join(current_parts))
                # Keep last part for overlap
                current_parts = [current_parts[-1]] if current_parts else []
                current_words = len(current_parts[0].split()) if current_parts else 0

            # Split the big paragraph
            sub_chunks = _split_long_paragraph(para, chunk_size, chunk_overlap)
            for sub in sub_chunks[:-1]:
                chunks_text.append(sub)
            # Last sub-chunk goes into buffer
            current_parts = [sub_chunks[-1]]
            current_words = len(sub_chunks[-1].split())
            continue

        # Would adding this paragraph exceed our limit?
        if current_words + para_words > chunk_size and current_parts:
            chunks_text.append("\n\n".join(current_parts))
            # Overlap: keep the last paragraph
            overlap_part = current_parts[-1]
            current_parts = [overlap_part, para]
            current_words = len(overlap_part.split()) + para_words
        else:
            current_parts.append(para)
            current_words += para_words

    # Flush remaining buffer
    if current_parts:
        chunks_text.append("\n\n".join(current_parts))

    # Filter tiny chunks and build Chunk objects
    valid_chunks = [c for c in chunks_text if len(c.split()) >= min_chunk_words]
    total = len(valid_chunks)

    result = []
    for i, text in enumerate(valid_chunks):
        chunk = Chunk(
            text=text,
            chunk_id=f"{source_slug}_{i:04d}",
            source=doc.source,
            source_type=doc.source_type,
            title=doc.title,
            chunk_index=i,
            total_chunks=total,
            created_at=doc.created_at,
            metadata={
                **doc.metadata,
                "chunk_index": i,
                "total_chunks": total,
            }
        )
        result.append(chunk)

    return result


def chunk_documents(
    docs: list[RawDocument],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    verbose: bool = True,
) -> list[Chunk]:
    """Chunk a list of documents and return all chunks."""
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if verbose:
            print(f"  📄 {doc.title}: {len(chunks)} chunks from {len(doc.content.split())} words")
        all_chunks.extend(chunks)
    return all_chunks
