"""Base tool interface for external API integrations."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    success: bool
    data: dict = field(default_factory=dict)
    error: str = ""

    def to_text(self) -> str:
        """Return a human-readable text summary of the result."""
        if not self.success:
            return f"工具调用失败: {self.error}"
        return str(self.data)


class BaseTool(ABC):
    """Abstract base class for external API tools."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            ToolResult with the outcome.
        """
        pass

    def get_schema(self) -> dict:
        """Return a JSON-schema-style description of this tool's parameters."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {},
        }
