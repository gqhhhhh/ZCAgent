"""Safety checker for cockpit commands.

安全检查引擎：根据驾驶状态（停车/行驶/高速）和预设规则，判断操作是否允许执行。
规则从 config/config.yaml 的 safety 部分加载，支持阻止列表和确认列表两种策略。
"""

import logging
import os
from dataclasses import dataclass, field

import yaml

from src.cockpit.domains import DomainType, IntentType, ParsedIntent

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """Result of a safety check on a parsed intent."""
    is_safe: bool
    requires_confirmation: bool = False
    blocked_reason: str = ""
    warnings: list[str] = field(default_factory=list)


class SafetyChecker:
    """Validates cockpit commands against safety rules."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = self._load_config()
        self.blocked_while_driving = set(config.get("blocked_while_driving", []))
        self.require_confirmation = set(config.get("require_confirmation", []))

    def _load_config(self) -> dict:
        """Load safety configuration."""
        config_paths = [
            os.path.join(os.path.dirname(__file__), "../../config/config.yaml"),
            "config/config.yaml",
        ]
        for path in config_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    full_config = yaml.safe_load(f)
                return full_config.get("safety", {})
        return {}

    def check(self, intent: ParsedIntent, driving_state: str = "parked") -> SafetyCheckResult:
        """Check if an intent is safe to execute given the current driving state.

        Args:
            intent: The parsed user intent.
            driving_state: Current driving state ('parked', 'driving', 'highway').

        Returns:
            SafetyCheckResult with safety assessment.
        """
        warnings = []

        # 安全类操作（如 SOS、ADAS）始终放行，拥有最高优先级
        if intent.domain == DomainType.SAFETY:
            if intent.intent_type == IntentType.EMERGENCY_CALL:
                logger.info("Safety pass: emergency_call (requires confirmation)")
                return SafetyCheckResult(
                    is_safe=True,
                    requires_confirmation=True,
                    warnings=["紧急呼叫将立即执行"],
                )
            logger.info("Safety pass: %s (safety domain)", intent.intent_type.value)
            return SafetyCheckResult(is_safe=True)

        # 行驶中检查是否存在被禁止的操作（如看视频、浏览网页）
        if driving_state in ("driving", "highway"):
            action_name = intent.intent_type.value
            if action_name in self.blocked_while_driving:
                logger.warning("Safety BLOCKED: '%s' while %s", action_name, driving_state)
                return SafetyCheckResult(
                    is_safe=False,
                    blocked_reason=f"操作 '{action_name}' 在行驶中被禁止",
                )

            # 某些操作在行驶/高速中需要用户二次确认后才执行
            check_key = action_name
            if driving_state == "highway" and intent.intent_type == IntentType.OPEN_WINDOW:
                check_key = "open_window_highway"
            if check_key in self.require_confirmation:
                logger.info("Safety CONFIRM required: '%s' while %s", action_name, driving_state)
                return SafetyCheckResult(
                    is_safe=True,
                    requires_confirmation=True,
                    warnings=[f"操作 '{action_name}' 需要确认"],
                )

            # Warn about complex operations while driving
            if intent.domain == DomainType.NAVIGATION and driving_state == "highway":
                warnings.append("高速行驶中，建议使用语音交互完成导航设置")

        return SafetyCheckResult(is_safe=True, warnings=warnings)
