"""Base agent class for the multi-agent system."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from an agent."""
    content: str
    intent_results: list[dict] = field(default_factory=list)
    task_results: list[dict] = field(default_factory=list)
    requires_confirmation: bool = False
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self, name: str, llm_client=None):
        self.name = name
        self.llm_client = llm_client

    @abstractmethod
    def process(self, user_input: str, context: dict | None = None) -> AgentResponse:
        """Process user input and return a response.

        Args:
            user_input: The user's natural language input.
            context: Optional context from memory and other agents.

        Returns:
            AgentResponse with results.
        """
        pass

    def _build_system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        return "You are an intelligent cockpit assistant."
