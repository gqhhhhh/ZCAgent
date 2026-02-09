"""Long-term memory with importance scoring and conflict resolution."""

import time
from dataclasses import dataclass, field


@dataclass
class LongTermItem:
    """An item in long-term memory."""
    key: str
    content: str
    category: str = "general"  # 'preference', 'fact', 'habit', 'general'
    importance: float = 0.5
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def access(self):
        """Record an access to this memory item."""
        self.access_count += 1
        self.updated_at = time.time()


class LongTermMemory:
    """Manages persistent user preferences and knowledge.

    Supports importance-based filtering, conflict resolution when
    new information contradicts existing memories, and LLM-based
    summarization for memory compression.
    """

    def __init__(self, importance_threshold: float = 0.7,
                 summary_trigger_count: int = 20, llm_client=None):
        self.importance_threshold = importance_threshold
        self.summary_trigger_count = summary_trigger_count
        self.llm_client = llm_client
        self._items: dict[str, LongTermItem] = {}
        self._pending_items: list[dict] = []

    def store(self, key: str, content: str, category: str = "general",
              importance: float = 0.5, metadata: dict | None = None) -> LongTermItem:
        """Store or update an item in long-term memory.

        If the key already exists, triggers conflict resolution.
        Only stores items above the importance threshold.
        """
        if importance < self.importance_threshold:
            # Track as pending; may be promoted later
            self._pending_items.append({
                "key": key, "content": content, "category": category,
                "importance": importance,
            })
            # Trigger summarization if too many pending items
            if len(self._pending_items) >= self.summary_trigger_count:
                self._summarize_pending()
            existing = self._items.get(key)
            if existing:
                return existing
            # Return a transient item (not stored)
            return LongTermItem(key=key, content=content, category=category,
                                importance=importance, metadata=metadata or {})

        if key in self._items:
            return self._resolve_conflict(key, content, importance, metadata)

        item = LongTermItem(
            key=key, content=content, category=category,
            importance=importance, metadata=metadata or {},
        )
        self._items[key] = item
        return item

    def retrieve(self, key: str) -> LongTermItem | None:
        """Retrieve an item by key and record the access."""
        item = self._items.get(key)
        if item:
            item.access()
        return item

    def search(self, query: str, category: str | None = None) -> list[LongTermItem]:
        """Search long-term memory by keyword and optional category."""
        query_lower = query.lower()
        results = []
        for item in self._items.values():
            if category and item.category != category:
                continue
            if query_lower in item.content.lower() or query_lower in item.key.lower():
                item.access()
                results.append(item)
        return sorted(results, key=lambda x: x.importance, reverse=True)

    def get_preferences(self) -> list[LongTermItem]:
        """Get all user preference items."""
        return [
            item for item in self._items.values()
            if item.category == "preference"
        ]

    def get_all(self) -> list[LongTermItem]:
        """Get all long-term memory items."""
        return list(self._items.values())

    def remove(self, key: str) -> bool:
        """Remove an item from long-term memory."""
        if key in self._items:
            del self._items[key]
            return True
        return False

    def clear(self):
        """Clear all long-term memory."""
        self._items.clear()
        self._pending_items.clear()

    def _resolve_conflict(self, key: str, new_content: str,
                          new_importance: float,
                          metadata: dict | None) -> LongTermItem:
        """Resolve conflict when new info contradicts existing memory.

        Strategy: If new importance >= existing importance, update.
        Otherwise, keep existing but record the conflict.
        """
        existing = self._items[key]

        if new_importance >= existing.importance:
            existing.metadata["previous_content"] = existing.content
            existing.content = new_content
            existing.importance = new_importance
            existing.updated_at = time.time()
            if metadata:
                existing.metadata.update(metadata)
            existing.metadata["conflict_resolved"] = True
        else:
            existing.metadata.setdefault("conflicting_info", [])
            existing.metadata["conflicting_info"].append({
                "content": new_content,
                "importance": new_importance,
                "timestamp": time.time(),
            })

        return existing

    def _summarize_pending(self):
        """Summarize pending items and promote important ones."""
        if not self._pending_items:
            return

        if self.llm_client:
            try:
                items_text = "\n".join(
                    f"- [{p['category']}] {p['content']}"
                    for p in self._pending_items
                )
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "从以下对话记录中提取重要的用户偏好和事实。"
                            "返回JSON数组，每项包含 key, content, category, importance(0-1)。"
                            "只返回importance > 0.7的项。"
                        ),
                    },
                    {"role": "user", "content": items_text},
                ]
                results = self.llm_client.generate_json(messages)
                if isinstance(results, list):
                    for r in results:
                        self.store(
                            key=r.get("key", ""),
                            content=r.get("content", ""),
                            category=r.get("category", "general"),
                            importance=r.get("importance", 0.8),
                        )
            except Exception as e:
                logger.debug("Failed to summarize pending items: %s", e)

        self._pending_items.clear()

    @property
    def size(self) -> int:
        return len(self._items)
