"""Task graph for modeling task dependencies and parallel execution."""

import time
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    """Status of a task in the graph."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 100  # Safety-related
    HIGH = 80       # Navigation
    MEDIUM = 60     # Vehicle settings
    NORMAL = 50     # Music, general
    LOW = 30        # Background tasks


@dataclass
class Task:
    """A task node in the task graph."""
    task_id: str
    name: str
    domain: str
    action: str
    params: dict = field(default_factory=dict)
    priority: int = 50
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict | None = None
    error: str | None = None

    def is_ready(self, completed_tasks: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.dependencies)

    def mark_running(self):
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()

    def mark_completed(self, result: dict | None = None):
        self.status = TaskStatus.COMPLETED
        self.completed_at = time.time()
        self.result = result

    def mark_failed(self, error: str = ""):
        self.status = TaskStatus.FAILED
        self.completed_at = time.time()
        self.error = error

    def mark_cancelled(self):
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()


class TaskGraph:
    """Directed acyclic graph for task dependency management.

    Supports parallel execution of independent tasks and sequential
    execution of dependent tasks.
    """

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._completed: set[str] = set()

    def add_task(self, task: Task) -> Task:
        """Add a task to the graph."""
        self._tasks[task.task_id] = task
        return task

    def add_dependency(self, task_id: str, depends_on: str):
        """Add a dependency between tasks."""
        if task_id in self._tasks and depends_on in self._tasks:
            if depends_on not in self._tasks[task_id].dependencies:
                self._tasks[task_id].dependencies.append(depends_on)

    def get_ready_tasks(self) -> list[Task]:
        """Get all tasks that are ready to execute (dependencies satisfied).

        Returns tasks sorted by priority (highest first).
        """
        ready = []
        for task in self._tasks.values():
            if task.status == TaskStatus.PENDING and task.is_ready(self._completed):
                task.status = TaskStatus.READY
                ready.append(task)
            elif task.status == TaskStatus.READY:
                ready.append(task)
        return sorted(ready, key=lambda t: t.priority, reverse=True)

    def complete_task(self, task_id: str, result: dict | None = None):
        """Mark a task as completed."""
        if task_id in self._tasks:
            self._tasks[task_id].mark_completed(result)
            self._completed.add(task_id)

    def fail_task(self, task_id: str, error: str = ""):
        """Mark a task as failed and cancel dependent tasks."""
        if task_id in self._tasks:
            self._tasks[task_id].mark_failed(error)
            # Cancel tasks that depend on the failed task
            for other in self._tasks.values():
                if task_id in other.dependencies and other.status in (
                    TaskStatus.PENDING, TaskStatus.READY
                ):
                    other.mark_cancelled()

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[Task]:
        """Get all tasks in the graph."""
        return list(self._tasks.values())

    def is_complete(self) -> bool:
        """Check if all tasks are in a terminal state."""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            for t in self._tasks.values()
        )

    def get_execution_order(self) -> list[list[str]]:
        """Get tasks grouped by execution wave (parallel groups).

        Returns a list of lists, where each inner list contains
        task IDs that can be executed in parallel.
        """
        waves: list[list[str]] = []
        completed = set()
        remaining = {tid for tid, t in self._tasks.items()
                     if t.status not in (TaskStatus.CANCELLED, TaskStatus.FAILED)}

        while remaining:
            wave = []
            for tid in list(remaining):
                task = self._tasks[tid]
                if task.is_ready(completed):
                    wave.append(tid)
            if not wave:
                break  # Circular dependency or all remaining blocked
            waves.append(sorted(wave, key=lambda t: self._tasks[t].priority, reverse=True))
            completed.update(wave)
            remaining -= set(wave)

        return waves

    def clear(self):
        """Clear the task graph."""
        self._tasks.clear()
        self._completed.clear()
