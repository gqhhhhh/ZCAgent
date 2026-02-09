"""Working memory for current conversation context and active tasks."""

import time
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class MemoryItem:
    """A single item in working memory."""
    key: str
    content: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    metadata: dict = field(default_factory=dict)


class WorkingMemory:
    """Manages current conversation context with limited capacity.

    Working memory holds the most recent and relevant items for the
    current interaction, similar to human working memory.
    """

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self._items: OrderedDict[str, MemoryItem] = OrderedDict()

    def add(self, key: str, content: str, importance: float = 0.5,
            metadata: dict | None = None) -> MemoryItem:
        """Add or update an item in working memory.

        If capacity is exceeded, the least important item is evicted.
        """
        item = MemoryItem(
            key=key,
            content=content,
            importance=importance,
            metadata=metadata or {},
        )

        if key in self._items:
            del self._items[key]
        self._items[key] = item

        # Evict least important item if over capacity
        while len(self._items) > self.capacity:
            self._evict_least_important()

        return item

    def get(self, key: str) -> MemoryItem | None:
        """Retrieve an item by key."""
        return self._items.get(key)

    def get_all(self) -> list[MemoryItem]:
        """Get all items in working memory, ordered by recency."""
        return list(self._items.values())

    def remove(self, key: str) -> bool:
        """Remove an item from working memory."""
        if key in self._items:
            del self._items[key]
            return True
        return False

    def clear(self):
        """Clear all working memory."""
        self._items.clear()

    def get_context_summary(self) -> str:
        """Get a text summary of current working memory contents."""
        if not self._items:
            return ""
        parts = []
        for item in self._items.values():
            parts.append(f"[{item.key}]: {item.content}")
        return "\n".join(parts)

    def _evict_least_important(self):
        """Remove the least important item from working memory."""
        if not self._items:
            return
        min_key = min(self._items, key=lambda k: self._items[k].importance)
        del self._items[min_key]

    @property
    def size(self) -> int:
        return len(self._items)
