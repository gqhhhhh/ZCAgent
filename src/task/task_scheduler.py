"""Priority-based task scheduler with safety constraints."""

import logging
from dataclasses import dataclass

from src.task.task_graph import Task, TaskGraph, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """Configuration for the task scheduler."""
    max_parallel_tasks: int = 5
    safety_priority: int = TaskPriority.CRITICAL.value
    navigation_priority: int = TaskPriority.HIGH.value
    phone_priority: int = 70
    vehicle_setting_priority: int = TaskPriority.MEDIUM.value
    music_priority: int = TaskPriority.NORMAL.value

    def get_domain_priority(self, domain: str) -> int:
        """Get priority for a domain."""
        priority_map = {
            "safety": self.safety_priority,
            "navigation": self.navigation_priority,
            "phone": self.phone_priority,
            "vehicle_setting": self.vehicle_setting_priority,
            "music": self.music_priority,
        }
        return priority_map.get(domain, TaskPriority.NORMAL.value)


class TaskScheduler:
    """Schedules tasks based on priority and resource constraints.

    Implements priority-based scheduling with support for:
    - Domain-based priority assignment
    - Safety task preemption
    - Parallel execution limits
    - Task cancellation for conflicting operations
    """

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig()
        self.graph = TaskGraph()
        self._running_tasks: set[str] = set()

    def submit_task(self, task: Task) -> Task:
        """Submit a task for scheduling.

        Assigns priority based on domain if not explicitly set,
        and checks for conflicts with running tasks.
        """
        # Assign domain-based priority if using default
        if task.priority == TaskPriority.NORMAL.value:
            task.priority = self.config.get_domain_priority(task.domain)

        # Check for conflicts
        self._handle_conflicts(task)

        self.graph.add_task(task)
        logger.info("Task submitted: %s (priority=%d)", task.task_id, task.priority)
        return task

    def get_next_tasks(self) -> list[Task]:
        """Get the next batch of tasks to execute.

        Returns tasks respecting parallel execution limits and priorities.
        """
        available_slots = self.config.max_parallel_tasks - len(self._running_tasks)
        if available_slots <= 0:
            return []

        ready_tasks = self.graph.get_ready_tasks()

        # Safety tasks always get priority - preempt if necessary
        safety_tasks = [t for t in ready_tasks if t.domain == "safety"]
        if safety_tasks:
            return safety_tasks[:available_slots]

        return ready_tasks[:available_slots]

    def start_task(self, task_id: str):
        """Mark a task as started."""
        task = self.graph.get_task(task_id)
        if task:
            task.mark_running()
            self._running_tasks.add(task_id)

    def complete_task(self, task_id: str, result: dict | None = None):
        """Mark a task as completed."""
        self.graph.complete_task(task_id, result)
        self._running_tasks.discard(task_id)

    def fail_task(self, task_id: str, error: str = ""):
        """Mark a task as failed."""
        self.graph.fail_task(task_id, error)
        self._running_tasks.discard(task_id)

    def _handle_conflicts(self, new_task: Task):
        """Handle conflicts between new task and existing tasks.

        For example, a new navigation task cancels the previous navigation.
        """
        conflicting_domains = {"navigation", "music"}
        if new_task.domain in conflicting_domains:
            for task in self.graph.get_all_tasks():
                if (task.domain == new_task.domain
                    and task.status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RUNNING)
                    and task.task_id != new_task.task_id):
                    task.mark_cancelled()
                    self._running_tasks.discard(task.task_id)
                    logger.info("Cancelled conflicting task: %s", task.task_id)

    def is_complete(self) -> bool:
        """Check if all scheduled tasks are done."""
        return self.graph.is_complete()

    def get_status(self) -> dict:
        """Get scheduler status summary."""
        tasks = self.graph.get_all_tasks()
        return {
            "total_tasks": len(tasks),
            "running": len(self._running_tasks),
            "pending": sum(1 for t in tasks if t.status in (TaskStatus.PENDING, TaskStatus.READY)),
            "completed": sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in tasks if t.status == TaskStatus.FAILED),
            "cancelled": sum(1 for t in tasks if t.status == TaskStatus.CANCELLED),
        }
