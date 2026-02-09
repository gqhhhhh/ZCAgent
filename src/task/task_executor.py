"""Task executor for processing scheduled tasks."""

import logging
import uuid

from src.cockpit.domains import DomainType, IntentType, ParsedIntent
from src.task.task_graph import Task, TaskPriority

logger = logging.getLogger(__name__)

# Domain to priority mapping
DOMAIN_PRIORITY = {
    DomainType.SAFETY: TaskPriority.CRITICAL.value,
    DomainType.NAVIGATION: TaskPriority.HIGH.value,
    DomainType.PHONE: 70,
    DomainType.VEHICLE_SETTING: TaskPriority.MEDIUM.value,
    DomainType.MUSIC: TaskPriority.NORMAL.value,
    DomainType.GENERAL: TaskPriority.LOW.value,
}


class TaskExecutor:
    """Converts parsed intents into executable tasks and processes them.

    Bridges the intent parsing system with the task scheduling system.
    """

    def __init__(self, scheduler=None):
        self.scheduler = scheduler
        self._handlers: dict[str, object] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default task handlers for each domain."""
        self._handlers = {
            "navigate_to": self._handle_navigation,
            "search_poi": self._handle_navigation,
            "make_call": self._handle_phone,
            "answer_call": self._handle_phone,
            "reject_call": self._handle_phone,
            "play_music": self._handle_music,
            "pause_music": self._handle_music,
            "next_track": self._handle_music,
            "adjust_volume": self._handle_vehicle,
            "set_temperature": self._handle_vehicle,
            "open_window": self._handle_vehicle,
            "close_window": self._handle_vehicle,
            "emergency_call": self._handle_safety,
            "adas_control": self._handle_safety,
        }

    def register_handler(self, action: str, handler):
        """Register a custom handler for a task action."""
        self._handlers[action] = handler

    def intent_to_task(self, intent: ParsedIntent) -> Task:
        """Convert a parsed intent into an executable task."""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        priority = DOMAIN_PRIORITY.get(intent.domain, TaskPriority.NORMAL.value)

        return Task(
            task_id=task_id,
            name=f"{intent.domain.value}:{intent.intent_type.value}",
            domain=intent.domain.value,
            action=intent.intent_type.value,
            params=intent.slots.copy(),
            priority=priority,
        )

    def execute(self, task: Task) -> dict:
        """Execute a single task and return the result.

        Args:
            task: The task to execute.

        Returns:
            Dict with execution result.
        """
        handler = self._handlers.get(task.action)
        if handler:
            return handler(task)
        return {
            "status": "success",
            "message": f"Task '{task.name}' executed (no specific handler)",
            "action": task.action,
            "params": task.params,
        }

    def _handle_navigation(self, task: Task) -> dict:
        """Handle navigation tasks."""
        destination = task.params.get("destination", "未指定目的地")
        return {
            "status": "success",
            "domain": "navigation",
            "action": task.action,
            "message": f"导航到: {destination}",
            "params": task.params,
        }

    def _handle_phone(self, task: Task) -> dict:
        """Handle phone tasks."""
        contact = task.params.get("contact", "未知联系人")
        return {
            "status": "success",
            "domain": "phone",
            "action": task.action,
            "message": f"电话操作: {task.action} -> {contact}",
            "params": task.params,
        }

    def _handle_music(self, task: Task) -> dict:
        """Handle music tasks."""
        query = task.params.get("query", "")
        return {
            "status": "success",
            "domain": "music",
            "action": task.action,
            "message": f"音乐操作: {task.action}" + (f" -> {query}" if query else ""),
            "params": task.params,
        }

    def _handle_vehicle(self, task: Task) -> dict:
        """Handle vehicle setting tasks."""
        return {
            "status": "success",
            "domain": "vehicle_setting",
            "action": task.action,
            "message": f"车辆设置: {task.action}",
            "params": task.params,
        }

    def _handle_safety(self, task: Task) -> dict:
        """Handle safety-critical tasks."""
        return {
            "status": "success",
            "domain": "safety",
            "action": task.action,
            "message": f"安全操作: {task.action}",
            "params": task.params,
            "priority": "critical",
        }
