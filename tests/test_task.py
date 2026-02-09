"""Tests for the task graph and scheduler system."""

import pytest

from src.task.task_graph import Task, TaskGraph, TaskStatus, TaskPriority
from src.task.task_scheduler import TaskScheduler, SchedulerConfig
from src.task.task_executor import TaskExecutor
from src.cockpit.domains import DomainType, IntentType, ParsedIntent


class TestTaskGraph:
    def test_add_task(self):
        graph = TaskGraph()
        task = Task(task_id="t1", name="test", domain="navigation", action="navigate_to")
        graph.add_task(task)
        assert graph.get_task("t1") is not None

    def test_get_ready_tasks(self):
        graph = TaskGraph()
        t1 = Task(task_id="t1", name="nav", domain="navigation", action="navigate_to", priority=80)
        t2 = Task(task_id="t2", name="music", domain="music", action="play_music", priority=50)
        graph.add_task(t1)
        graph.add_task(t2)
        ready = graph.get_ready_tasks()
        assert len(ready) == 2
        # Higher priority first
        assert ready[0].task_id == "t1"

    def test_dependencies(self):
        graph = TaskGraph()
        t1 = Task(task_id="t1", name="first", domain="safety", action="safety_check")
        t2 = Task(task_id="t2", name="second", domain="navigation", action="navigate_to",
                  dependencies=["t1"])
        graph.add_task(t1)
        graph.add_task(t2)
        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == "t1"

        # Complete t1, now t2 should be ready
        graph.complete_task("t1")
        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == "t2"

    def test_fail_cascades(self):
        graph = TaskGraph()
        t1 = Task(task_id="t1", name="first", domain="safety", action="check")
        t2 = Task(task_id="t2", name="second", domain="nav", action="nav",
                  dependencies=["t1"])
        graph.add_task(t1)
        graph.add_task(t2)
        graph.fail_task("t1", "error")
        t2_status = graph.get_task("t2")
        assert t2_status.status == TaskStatus.CANCELLED

    def test_execution_order(self):
        graph = TaskGraph()
        t1 = Task(task_id="t1", name="a", domain="safety", action="check")
        t2 = Task(task_id="t2", name="b", domain="nav", action="nav", dependencies=["t1"])
        t3 = Task(task_id="t3", name="c", domain="music", action="play", dependencies=["t1"])
        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)
        waves = graph.get_execution_order()
        assert len(waves) == 2
        assert waves[0] == ["t1"]
        assert set(waves[1]) == {"t2", "t3"}

    def test_is_complete(self):
        graph = TaskGraph()
        t1 = Task(task_id="t1", name="a", domain="nav", action="nav")
        graph.add_task(t1)
        assert not graph.is_complete()
        graph.complete_task("t1")
        assert graph.is_complete()


class TestTaskScheduler:
    def test_submit_and_get_next(self):
        scheduler = TaskScheduler()
        task = Task(task_id="t1", name="nav", domain="navigation", action="navigate_to")
        scheduler.submit_task(task)
        next_tasks = scheduler.get_next_tasks()
        assert len(next_tasks) == 1

    def test_safety_preemption(self):
        scheduler = TaskScheduler()
        t1 = Task(task_id="t1", name="music", domain="music", action="play")
        t2 = Task(task_id="t2", name="sos", domain="safety", action="emergency")
        scheduler.submit_task(t1)
        scheduler.submit_task(t2)
        next_tasks = scheduler.get_next_tasks()
        assert next_tasks[0].task_id == "t2"

    def test_conflict_cancellation(self):
        scheduler = TaskScheduler()
        t1 = Task(task_id="t1", name="nav1", domain="navigation", action="navigate_to")
        scheduler.submit_task(t1)
        t2 = Task(task_id="t2", name="nav2", domain="navigation", action="navigate_to")
        scheduler.submit_task(t2)
        # First nav should be cancelled
        assert scheduler.graph.get_task("t1").status == TaskStatus.CANCELLED

    def test_parallel_limit(self):
        config = SchedulerConfig(max_parallel_tasks=2)
        scheduler = TaskScheduler(config=config)
        for i in range(5):
            task = Task(task_id=f"t{i}", name=f"task{i}", domain="general", action="chat")
            scheduler.submit_task(task)
        # Start 2 tasks
        next_tasks = scheduler.get_next_tasks()
        for t in next_tasks:
            scheduler.start_task(t.task_id)
        # Should not return more tasks until some complete
        assert len(scheduler.get_next_tasks()) == 0

    def test_status(self):
        scheduler = TaskScheduler()
        task = Task(task_id="t1", name="test", domain="general", action="chat")
        scheduler.submit_task(task)
        status = scheduler.get_status()
        assert status["total_tasks"] == 1
        assert status["pending"] == 1


class TestTaskExecutor:
    def test_intent_to_task(self):
        executor = TaskExecutor()
        intent = ParsedIntent(
            intent_type=IntentType.NAVIGATE_TO,
            domain=DomainType.NAVIGATION,
            confidence=0.9,
            slots={"destination": "天安门"},
        )
        task = executor.intent_to_task(intent)
        assert task.domain == "navigation"
        assert task.action == "navigate_to"
        assert task.params["destination"] == "天安门"

    def test_execute_navigation(self):
        executor = TaskExecutor()
        task = Task(task_id="t1", name="nav", domain="navigation",
                   action="navigate_to", params={"destination": "北京"})
        result = executor.execute(task)
        assert result["status"] == "success"
        assert "北京" in result["message"]

    def test_execute_music(self):
        executor = TaskExecutor()
        task = Task(task_id="t1", name="music", domain="music",
                   action="play_music", params={"query": "晴天"})
        result = executor.execute(task)
        assert result["status"] == "success"

    def test_execute_safety(self):
        executor = TaskExecutor()
        task = Task(task_id="t1", name="sos", domain="safety",
                   action="emergency_call")
        result = executor.execute(task)
        assert result["priority"] == "critical"

    def test_custom_handler(self):
        executor = TaskExecutor()
        executor.register_handler("custom_action", lambda t: {"status": "custom_ok"})
        task = Task(task_id="t1", name="custom", domain="general", action="custom_action")
        result = executor.execute(task)
        assert result["status"] == "custom_ok"
