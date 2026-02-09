"""Tests for the layered memory system."""

import time
import pytest

from src.memory.working_memory import WorkingMemory
from src.memory.short_term_memory import ShortTermMemory
from src.memory.long_term_memory import LongTermMemory
from src.memory.memory_manager import MemoryManager


class TestWorkingMemory:
    def test_add_and_get(self):
        wm = WorkingMemory(capacity=5)
        wm.add("key1", "content1", importance=0.8)
        item = wm.get("key1")
        assert item is not None
        assert item.content == "content1"
        assert item.importance == 0.8

    def test_capacity_eviction(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a", "low", importance=0.1)
        wm.add("b", "mid", importance=0.5)
        wm.add("c", "high", importance=0.9)
        wm.add("d", "new", importance=0.6)
        # Least important ("a") should be evicted
        assert wm.get("a") is None
        assert wm.size == 3

    def test_update_existing(self):
        wm = WorkingMemory(capacity=5)
        wm.add("key1", "old", importance=0.5)
        wm.add("key1", "new", importance=0.8)
        item = wm.get("key1")
        assert item.content == "new"
        assert wm.size == 1

    def test_remove(self):
        wm = WorkingMemory(capacity=5)
        wm.add("key1", "content")
        assert wm.remove("key1") is True
        assert wm.get("key1") is None
        assert wm.remove("nonexistent") is False

    def test_clear(self):
        wm = WorkingMemory(capacity=5)
        wm.add("key1", "content1")
        wm.add("key2", "content2")
        wm.clear()
        assert wm.size == 0

    def test_context_summary(self):
        wm = WorkingMemory(capacity=5)
        wm.add("nav", "navigating to Beijing")
        summary = wm.get_context_summary()
        assert "nav" in summary
        assert "Beijing" in summary


class TestShortTermMemory:
    def test_add_and_get_recent(self):
        stm = ShortTermMemory(ttl_seconds=300)
        stm.add("Hello", role="user")
        stm.add("Hi there", role="assistant")
        recent = stm.get_recent(10)
        assert len(recent) == 2
        assert recent[0].content == "Hello"

    def test_get_messages(self):
        stm = ShortTermMemory(ttl_seconds=300)
        stm.add("What's the weather?", role="user")
        stm.add("It's sunny.", role="assistant")
        messages = stm.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["content"] == "It's sunny."

    def test_ttl_expiry(self):
        stm = ShortTermMemory(ttl_seconds=0.1)
        stm.add("old message", role="user")
        time.sleep(0.2)
        assert stm.size == 0

    def test_search(self):
        stm = ShortTermMemory(ttl_seconds=300)
        stm.add("Navigate to Beijing", role="user")
        stm.add("Playing music", role="user")
        results = stm.search("Beijing")
        assert len(results) == 1
        assert "Beijing" in results[0].content


class TestLongTermMemory:
    def test_store_above_threshold(self):
        ltm = LongTermMemory(importance_threshold=0.7)
        ltm.store("pref1", "likes jazz", category="preference", importance=0.8)
        item = ltm.retrieve("pref1")
        assert item is not None
        assert item.content == "likes jazz"

    def test_store_below_threshold(self):
        ltm = LongTermMemory(importance_threshold=0.7)
        ltm.store("low", "unimportant", importance=0.3)
        assert ltm.retrieve("low") is None
        assert ltm.size == 0

    def test_conflict_resolution_higher_importance(self):
        ltm = LongTermMemory(importance_threshold=0.5)
        ltm.store("music", "likes rock", importance=0.6)
        ltm.store("music", "likes jazz", importance=0.8)
        item = ltm.retrieve("music")
        assert item.content == "likes jazz"

    def test_search(self):
        ltm = LongTermMemory(importance_threshold=0.5)
        ltm.store("pref_music", "prefers classical music", category="preference", importance=0.8)
        ltm.store("pref_temp", "prefers 22 degrees", category="preference", importance=0.7)
        results = ltm.search("music")
        assert len(results) == 1
        assert "classical" in results[0].content

    def test_get_preferences(self):
        ltm = LongTermMemory(importance_threshold=0.5)
        ltm.store("pref1", "likes jazz", category="preference", importance=0.8)
        ltm.store("fact1", "lives in Beijing", category="fact", importance=0.7)
        prefs = ltm.get_preferences()
        assert len(prefs) == 1
        assert prefs[0].category == "preference"


class TestMemoryManager:
    def test_add_messages(self):
        mm = MemoryManager()
        mm.add_user_message("导航到天安门")
        mm.add_assistant_message("正在导航")
        ctx = mm.get_context()
        assert len(ctx["recent_messages"]) == 2

    def test_store_preference(self):
        mm = MemoryManager()
        mm.store_preference("music_pref", "喜欢爵士乐")
        prefs = mm.long_term.get_preferences()
        assert len(prefs) == 1

    def test_search(self):
        mm = MemoryManager()
        mm.add_user_message("导航到天安门")
        results = mm.search("天安门")
        assert len(results["short_term"]) == 1

    def test_clear_session(self):
        mm = MemoryManager()
        mm.add_user_message("test")
        mm.clear_session()
        assert mm.working.size == 0
        assert mm.short_term.size == 0
