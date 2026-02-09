"""Short-term memory with TTL-based expiry."""

import time
from dataclasses import dataclass, field


@dataclass
class ShortTermItem:
    """An item in short-term memory with expiry."""
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    metadata: dict = field(default_factory=dict)

    def is_expired(self, ttl: float) -> bool:
        """Check if this item has expired based on TTL."""
        return (time.time() - self.timestamp) > ttl


class ShortTermMemory:
    """Manages recent conversation history with automatic expiry.

    Items expire after a configurable TTL (time-to-live) period.
    Provides conversation context for the agent system.
    """

    def __init__(self, ttl_seconds: float = 300):
        self.ttl_seconds = ttl_seconds
        self._items: list[ShortTermItem] = []

    def add(self, content: str, role: str = "user", importance: float = 0.5,
            metadata: dict | None = None) -> ShortTermItem:
        """Add a conversation turn to short-term memory."""
        self._cleanup_expired()
        item = ShortTermItem(
            content=content,
            role=role,
            importance=importance,
            metadata=metadata or {},
        )
        self._items.append(item)
        return item

    def get_recent(self, n: int = 10) -> list[ShortTermItem]:
        """Get the N most recent non-expired items."""
        self._cleanup_expired()
        return self._items[-n:]

    def get_messages(self, n: int = 10) -> list[dict[str, str]]:
        """Get recent items as message dicts for LLM input."""
        items = self.get_recent(n)
        return [{"role": item.role, "content": item.content} for item in items]

    def search(self, query: str) -> list[ShortTermItem]:
        """Simple keyword search in short-term memory."""
        self._cleanup_expired()
        query_lower = query.lower()
        return [
            item for item in self._items
            if query_lower in item.content.lower()
        ]

    def clear(self):
        """Clear all short-term memory."""
        self._items.clear()

    def _cleanup_expired(self):
        """Remove expired items."""
        self._items = [
            item for item in self._items
            if not item.is_expired(self.ttl_seconds)
        ]

    @property
    def size(self) -> int:
        self._cleanup_expired()
        return len(self._items)
