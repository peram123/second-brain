"""
ingest.py
---------
CLI tool to add memories from the terminal.
Usage:
  python ingest.py --file resume.pdf
  python ingest.py --dir ./my_notes/
  python ingest.py --url https://myblog.com/about
  python ingest.py --text "I grew up in Hyderabad and moved to New York in 2023"
"""

import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

load_dotenv()
console = Console()


def run(args):
    from ingestion.loaders import load_file, load_directory, load_url, load_raw_text
    from ingestion.chunker import chunk_documents
    from ingestion.embedder import embed_chunks
    from memory.store import MemoryStore

    console.print(Panel.fit("🧠 [bold cyan]Second Brain — Memory Ingestion[/bold cyan]"))

    # Load documents
    docs = []

    if args.file:
        for f in args.file:
            console.print(f"📄 Loading file: [cyan]{f}[/cyan]")
            docs.append(load_file(f))

    if args.dir:
        console.print(f"📁 Loading directory: [cyan]{args.dir}[/cyan]")
        docs.extend(load_directory(args.dir))

    if args.url:
        for url in args.url:
            console.print(f"🌐 Loading URL: [cyan]{url}[/cyan]")
            docs.append(load_url(url))

    if args.text:
        console.print("✏️  Loading raw text note")
        docs.append(load_raw_text(args.text, title=f"Note {len(docs)+1}"))

    if not docs:
        console.print("[red]No input provided. Use --file, --dir, --url, or --text[/red]")
        sys.exit(1)

    console.print(f"\n✓ Loaded [bold]{len(docs)}[/bold] document(s)")

    # Chunk
    console.print("\n✂️  Chunking documents...")
    chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=50, verbose=True)
    console.print(f"✓ Created [bold]{len(chunks)}[/bold] chunks")

    # Embed
    console.print("\n🧠 Embedding chunks...")
    pairs = embed_chunks(chunks)

    # Store
    console.print("\n💾 Storing in memory...")
    store = MemoryStore(
        db_path=os.getenv("QDRANT_PATH", "./data/qdrant_db"),
        collection_name=os.getenv("COLLECTION_NAME", "personal_memory"),
    )
    stored = store.add_chunks(pairs)

    console.print(Panel.fit(
        f"✅ [bold green]Done![/bold green]\n"
        f"Stored [bold]{stored}[/bold] new memory chunks\n"
        f"Total memories: [bold]{store.count()}[/bold]",
        title="Success"
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add personal data to your Second Brain")
    parser.add_argument("--file", nargs="+", help="Path to one or more files (PDF, DOCX, TXT, MD)")
    parser.add_argument("--dir",  type=str,   help="Path to a directory (loads all supported files)")
    parser.add_argument("--url",  nargs="+",  help="One or more URLs to scrape")
    parser.add_argument("--text", type=str,   help="Raw text string to add as a memory")
    args = parser.parse_args()
    run(args)
