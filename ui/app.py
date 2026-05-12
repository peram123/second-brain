"""
ui/app.py
---------
Streamlit chat interface with memory transparency panel.
The key differentiator: every answer shows WHICH memories were used and how confident.
"""

import os
import sys
import streamlit as st
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Second Brain — Personal AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .memory-card {
        background: #1e2a3a;
        border-left: 3px solid #4FC3D4;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.85em;
    }
    .memory-stale {
        border-left-color: #f0a500;
    }
    .confidence-high   { color: #4FC3D4; font-weight: bold; }
    .confidence-medium { color: #f0a500; font-weight: bold; }
    .confidence-low    { color: #ff6b6b; font-weight: bold; }
    .confidence-none   { color: #888;    font-weight: bold; }
    .score-bar {
        height: 6px;
        border-radius: 3px;
        background: #4FC3D4;
        margin: 4px 0;
    }
    .source-tag {
        background: #2d3f50;
        color: #4FC3D4;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def get_memory_store():
    from memory.store import MemoryStore
    return MemoryStore(
        db_path=os.getenv("QDRANT_PATH", "./data/qdrant_db"),
        collection_name=os.getenv("COLLECTION_NAME", "personal_memory"),
        staleness_days=int(os.getenv("STALENESS_DAYS", "365")),
    )

@st.cache_resource
def get_rag_pipeline(_store):
    from rag.pipeline import SecondBrainRAG
    return SecondBrainRAG(
        memory_store=_store,
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        llm_provider=os.getenv("LLM_PROVIDER", "groq"),
        llm_model=os.getenv("LLM_MODEL", "llama3-70b-8192"),
        persona_name=st.session_state.get("persona_name", "the user"),
        top_k=int(os.getenv("TOP_K_MEMORIES", "5")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.35")),
    )


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "persona_name" not in st.session_state:
    st.session_state.persona_name = "the user"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Second Brain")
    st.markdown("*Your Personal Language Model*")
    st.divider()

    # Persona name
    persona = st.text_input("Your name", value=st.session_state.persona_name,
                             help="The AI will answer as if it knows you personally")
    st.session_state.persona_name = persona

    st.divider()

    # Memory upload section
    st.markdown("### Add Memories")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="Your data stays local — nothing is sent to external servers except the LLM API"
    )

    url_input = st.text_input("Or paste a URL", placeholder="https://...")

    text_input = st.text_area("Or type a note", placeholder="Add any text to your memory...", height=100)

    if st.button("Save to Memory", use_container_width=True):
        with st.spinner("Processing and embedding..."):
            try:
                store = get_memory_store()
                from ingestion.loaders import load_raw_text, load_url
                from ingestion.chunker import chunk_documents
                from ingestion.embedder import embed_chunks

                docs = []

                # Handle file uploads
                if uploaded_files:
                    import tempfile
                    for uf in uploaded_files:
                        suffix = Path(uf.name).suffix
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uf.read())
                            tmp_path = tmp.name
                        from ingestion.loaders import load_file
                        docs.append(load_file(tmp_path))

                # Handle URL
                if url_input.strip():
                    docs.append(load_url(url_input.strip()))

                # Handle text note
                if text_input.strip():
                    docs.append(load_raw_text(text_input.strip(), title=f"Note {datetime.now().strftime('%Y-%m-%d %H:%M')}"))

                if docs:
                    chunks = chunk_documents(docs, verbose=False)
                    pairs = embed_chunks(chunks)
                    stored = store.add_chunks(pairs)
                    st.success(f"✓ Added {stored} memory chunks from {len(docs)} source(s)")
                    st.cache_resource.clear()
                else:
                    st.warning("Nothing to save — upload a file, enter a URL, or type a note.")

            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Memory stats
    st.markdown("### Memory Stats")
    try:
        store = get_memory_store()
        total = store.count()
        st.metric("Total Memory Chunks", total)

        if total > 0:
            sources = store.list_sources()
            st.markdown(f"**{len(sources)} source(s):**")
            for s in sources[:10]:
                st.markdown(f"- `{s['title']}` ({s['chunks']} chunks)")
    except Exception as e:
        st.caption(f"Store not ready: {e}")

    st.divider()
    st.caption(" Privacy-first: memories stored locally in Qdrant.\nOnly queries sent to LLM API.")


# ── Main chat area ────────────────────────────────────────────────────────────
col_chat, col_memory = st.columns([3, 2])

with col_chat:
    st.markdown("##  Ask Your Second Brain")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything about yourself..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching memories..."):
                try:
                    store = get_memory_store()
                    rag = get_rag_pipeline(store)
                    rag.persona_name = st.session_state.persona_name

                    response = rag.ask(prompt)
                    st.session_state.last_response = response

                    st.markdown(response.answer)

                    # Confidence badge
                    conf = response.confidence
                    conf_colors = {"high": "🟢", "medium": "🟡", "low": "🔴", "none": "⚫"}
                    st.caption(
                        f"{conf_colors.get(conf, '⚫')} Confidence: **{conf.upper()}** | "
                        f"Memories used: **{len(response.memories_used)}** of {response.total_memories_searched} | "
                        f"Model: `{response.model_used}`"
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Make sure your API key is set in .env and you have memories stored.")


# ── Memory Transparency Panel ─────────────────────────────────────────────────
with col_memory:
    st.markdown("## Memory Transparency")
    st.caption("See exactly which memories influenced the last answer")

    if st.session_state.last_response:
        resp = st.session_state.last_response
        memories = resp.memories_used

        if not memories:
            st.info("No memories matched this query above the confidence threshold.")
        else:
            for i, mem in enumerate(memories, 1):
                is_stale = mem.is_stale
                card_class = "memory-card memory-stale" if is_stale else "memory-card"

                # Score bar width
                bar_width = int(mem.score * 100)
                trust_bar = int(mem.trust_weight * 100)

                st.markdown(f"""
<div class="{card_class}">
  <div style="display:flex; justify-content:space-between; margin-bottom:6px">
    <span><strong>Memory {i}</strong> &nbsp;
      <span class="source-tag">{mem.source_type.upper()}</span>
      {"⚠️ Older memory" if is_stale else ""}
    </span>
    <span style="color:#aaa; font-size:0.85em">{mem.created_at.strftime('%Y-%m-%d')}</span>
  </div>
  <div style="color:#ccc; margin-bottom:8px; font-size:0.9em">
    📄 <strong>{mem.title}</strong>
    &nbsp;(chunk {mem.chunk_index+1}/{mem.total_chunks})
  </div>
  <div style="display:flex; gap:16px; margin-bottom:6px; font-size:0.82em; color:#aaa">
    <span>Relevance: <strong style="color:#4FC3D4">{mem.score:.0%}</strong></span>
    <span>Trust: <strong style="color:{'#f0a500' if is_stale else '#4FC3D4'}">{mem.trust_weight:.0%}</strong></span>
  </div>
  <div class="score-bar" style="width:{bar_width}%"></div>
  <div style="margin-top:10px; color:#d0e8f0; line-height:1.5; font-size:0.88em">
    {mem.text[:300]}{"..." if len(mem.text) > 300 else ""}
  </div>
</div>
""", unsafe_allow_html=True)

    else:
        st.markdown("""
<div class="memory-card">
  <p style="color:#888; text-align:center; padding:20px">
    Ask a question to see which memories<br>were used to generate the answer.
  </p>
</div>
""", unsafe_allow_html=True)

        # Example questions
        st.markdown("### Try asking:")
        examples = [
            "What are my career goals?",
            "What projects have I worked on?",
            "What do I know about machine learning?",
            "What are my skills and experience?",
            "What have I written about recently?",

        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=ex):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()
