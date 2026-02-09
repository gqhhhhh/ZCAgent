"""Unified memory manager orchestrating working, short-term, and long-term memory."""

import logging

from src.memory.working_memory import WorkingMemory
from src.memory.short_term_memory import ShortTermMemory
from src.memory.long_term_memory import LongTermMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """Orchestrates the three-layer memory system.

    Manages information flow between working memory (active context),
    short-term memory (recent conversation), and long-term memory
    (persistent preferences and knowledge).
    """

    def __init__(self, config: dict | None = None, llm_client=None):
        config = config or {}
        self.working = WorkingMemory(
            capacity=config.get("working_memory_capacity", 10)
        )
        self.short_term = ShortTermMemory(
            ttl_seconds=config.get("short_term_ttl_seconds", 300)
        )
        self.long_term = LongTermMemory(
            importance_threshold=config.get("long_term_importance_threshold", 0.7),
            summary_trigger_count=config.get("summary_trigger_count", 20),
            llm_client=llm_client,
        )

    def add_user_message(self, content: str, metadata: dict | None = None):
        """Record a user message across memory layers."""
        # Add to short-term memory
        self.short_term.add(content, role="user", metadata=metadata)
        # Update working memory with latest user input
        self.working.add("last_user_input", content, importance=0.8, metadata=metadata)

    def add_assistant_message(self, content: str, metadata: dict | None = None):
        """Record an assistant response across memory layers."""
        self.short_term.add(content, role="assistant", metadata=metadata)
        self.working.add("last_assistant_response", content, importance=0.6,
                         metadata=metadata)

    def store_preference(self, key: str, content: str, importance: float = 0.8):
        """Store a user preference in long-term memory."""
        self.long_term.store(key, content, category="preference",
                            importance=importance)

    def store_fact(self, key: str, content: str, importance: float = 0.75):
        """Store a factual piece of information in long-term memory."""
        self.long_term.store(key, content, category="fact",
                            importance=importance)

    def get_context(self, n_recent: int = 5) -> dict:
        """Get unified context from all memory layers.

        Returns:
            Dict with working memory context, recent messages,
            and relevant long-term memories.
        """
        return {
            "working_memory": self.working.get_context_summary(),
            "recent_messages": self.short_term.get_messages(n_recent),
            "preferences": [
                {"key": p.key, "content": p.content}
                for p in self.long_term.get_preferences()
            ],
        }

    def search(self, query: str) -> dict:
        """Search across all memory layers."""
        return {
            "short_term": [
                {"content": item.content, "role": item.role}
                for item in self.short_term.search(query)
            ],
            "long_term": [
                {"key": item.key, "content": item.content,
                 "category": item.category}
                for item in self.long_term.search(query)
            ],
        }

    def clear_session(self):
        """Clear working and short-term memory (e.g., new session)."""
        self.working.clear()
        self.short_term.clear()

    def clear_all(self):
        """Clear all memory layers."""
        self.working.clear()
        self.short_term.clear()
        self.long_term.clear()
