"""
rag/pipeline.py
---------------
The core RAG engine: retrieve relevant memories → build prompt → generate answer.
Connects the memory store to the LLM and returns answers with full transparency.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from memory.store import MemoryStore, MemoryResult
from ingestion.embedder import embed_query


@dataclass
class RAGResponse:
    """Full response including the answer and the memories that produced it."""
    answer: str
    memories_used: list[MemoryResult]
    query: str
    total_memories_searched: int
    generated_at: datetime = field(default_factory=datetime.now)
    confidence: str = "high"        # high / medium / low / none
    model_used: str = ""


class SecondBrainRAG:
    """
    The main RAG pipeline for Second Brain.
    Retrieves personal memories and generates answers grounded in them.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-20250514",
        persona_name: str = "the user",
        top_k: int = 5,
        similarity_threshold: float = 0.35,
    ):
        self.memory_store = memory_store
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.persona_name = persona_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # Lazy-init LLM client
        self._llm_client = None

    def _get_llm_client(self):
        """Initialize LLM client on first use."""
        if self._llm_client is not None:
            return self._llm_client

        if self.llm_provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not set in environment")
                self._llm_client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic not installed. Run: pip install anthropic")

        elif self.llm_provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set in environment")
                self._llm_client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai not installed. Run: pip install openai")
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

        return self._llm_client

    def _build_system_prompt(self) -> str:
        return f"""You are a personal AI assistant for {self.persona_name}.
Your job is to answer questions based ONLY on the personal memories provided below.
These memories come from {self.persona_name}'s own documents, notes, and writings.

Rules:
- Answer as if you know {self.persona_name} deeply, using their own words and context.
- ONLY use information from the provided memories. Do not add outside knowledge.
- If the memories don't contain enough information to answer, say so clearly.
- Be specific and cite which memory/document the information came from when helpful.
- If memories contradict each other, acknowledge both perspectives.
- Keep answers concise but complete."""

    def _build_user_prompt(self, query: str, memories: list[MemoryResult]) -> str:
        if not memories:
            return f"""No relevant memories found for this query.

Query: {query}

Please let the user know you couldn't find relevant information in their personal memory store."""

        memory_blocks = []
        for i, mem in enumerate(memories, 1):
            age_note = " [OLDER MEMORY]" if mem.is_stale else ""
            block = (
                f"[Memory {i} | Source: {mem.title} | "
                f"Relevance: {mem.score:.0%} | Trust: {mem.trust_weight:.0%}{age_note}]\n"
                f"{mem.text}"
            )
            memory_blocks.append(block)

        memories_text = "\n\n---\n\n".join(memory_blocks)

        return f"""Personal memories retrieved for this query:

{memories_text}

---

Question: {query}

Answer based only on the memories above:"""

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the configured LLM and return the response text."""
        client = self._get_llm_client()

        if self.llm_provider == "anthropic":
            response = client.messages.create(
                model=self.llm_model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        elif self.llm_provider == "openai":
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=1024,
            )
            return response.choices[0].message.content

    def _determine_confidence(self, memories: list[MemoryResult]) -> str:
        """Estimate answer confidence based on memory scores."""
        if not memories:
            return "none"
        avg_score = sum(m.score for m in memories) / len(memories)
        top_score = memories[0].score if memories else 0

        if top_score >= 0.75 and avg_score >= 0.6:
            return "high"
        elif top_score >= 0.55:
            return "medium"
        elif top_score >= 0.35:
            return "low"
        else:
            return "none"

    def ask(self, query: str) -> RAGResponse:
        """
        Main entry point: ask a question, get an answer grounded in personal memories.
        """
        # 1. Embed the query
        query_embedding = embed_query(query, model_name=self.embedding_model)

        # 2. Retrieve relevant memories
        memories = self.memory_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            score_threshold=self.similarity_threshold,
        )

        # 3. Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, memories)

        # 4. Generate answer
        answer = self._call_llm(system_prompt, user_prompt)

        # 5. Assess confidence
        confidence = self._determine_confidence(memories)

        return RAGResponse(
            answer=answer,
            memories_used=memories,
            query=query,
            total_memories_searched=self.memory_store.count(),
            confidence=confidence,
            model_used=f"{self.llm_provider}/{self.llm_model}",
        )
