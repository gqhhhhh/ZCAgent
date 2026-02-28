"""Base tool interface for external API integrations.

外部工具抽象基类：定义 run() 调用接口和 ToolResult 统一返回结构。
所有外部 API 工具（高德地图、网页搜索等）继承此基类。
"""

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
