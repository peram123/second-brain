"""
rag/pipeline.py
---------------
RAG pipeline supporting Groq, Anthropic, and OpenAI.
Default: Groq (free, fast)
"""
 
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
 
from memory.store import MemoryStore, MemoryResult
from ingestion.embedder import embed_query
 
 
@dataclass
class RAGResponse:
    answer: str
    memories_used: list[MemoryResult]
    query: str
    total_memories_searched: int
    generated_at: datetime = field(default_factory=datetime.now)
    confidence: str = "high"
    model_used: str = ""
 
 
class SecondBrainRAG:
 
    def __init__(
        self,
        memory_store: MemoryStore,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "groq",
        llm_model: str = "llama3-70b-8192",
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
        self._llm_client = None
 
    def _get_llm_client(self):
        if self._llm_client is not None:
            return self._llm_client
 
        if self.llm_provider == "groq":
            try:
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not set in .env")
                self._llm_client = Groq(api_key=api_key)
            except ImportError:
                raise ImportError("groq not installed. Run: pip install groq")
 
        elif self.llm_provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not set in .env")
                self._llm_client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic not installed.")
 
        elif self.llm_provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set in .env")
                self._llm_client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai not installed.")
 
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
 
        return self._llm_client
 
    def _build_system_prompt(self) -> str:
        return f"""You are a personal AI assistant for {self.persona_name}.
Answer questions based ONLY on the personal memories provided.
These memories come from {self.persona_name}'s own documents, notes, and writings.
- Answer as if you know {self.persona_name} deeply.
- ONLY use information from the provided memories.
- If memories don't contain enough info, say so clearly.
- Keep answers concise but complete."""
 
    def _build_user_prompt(self, query: str, memories: list[MemoryResult]) -> str:
        if not memories:
            return f"No relevant memories found for: {query}\nTell the user no info was found."
 
        memory_blocks = []
        for i, mem in enumerate(memories, 1):
            age_note = " [OLDER MEMORY]" if mem.is_stale else ""
            block = (
                f"[Memory {i} | Source: {mem.title} | "
                f"Relevance: {mem.score:.0%}{age_note}]\n{mem.text}"
            )
            memory_blocks.append(block)
 
        memories_text = "\n\n---\n\n".join(memory_blocks)
        return f"Personal memories:\n\n{memories_text}\n\n---\n\nQuestion: {query}\n\nAnswer:"
 
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_llm_client()
 
        if self.llm_provider == "groq":
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
            )
            return response.choices[0].message.content
 
        elif self.llm_provider == "anthropic":
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
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
            )
            return response.choices[0].message.content
 
    def _determine_confidence(self, memories: list[MemoryResult]) -> str:
        if not memories:
            return "none"
        top_score = memories[0].score if memories else 0
        if top_score >= 0.75:
            return "high"
        elif top_score >= 0.55:
            return "medium"
        elif top_score >= 0.35:
            return "low"
        return "none"
 
    def ask(self, query: str) -> RAGResponse:
        query_embedding = embed_query(query, model_name=self.embedding_model)
        memories = self.memory_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            score_threshold=self.similarity_threshold,
        )
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, memories)
        answer = self._call_llm(system_prompt, user_prompt)
        confidence = self._determine_confidence(memories)
 
        return RAGResponse(
            answer=answer,
            memories_used=memories,
            query=query,
            total_memories_searched=self.memory_store.count(),
            confidence=confidence,
            model_used=f"{self.llm_provider}/{self.llm_model}",
        )
 