"""Safety checker for cockpit commands."""

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

        # Safety domain commands always get highest priority
        if intent.domain == DomainType.SAFETY:
            if intent.intent_type == IntentType.EMERGENCY_CALL:
                return SafetyCheckResult(
                    is_safe=True,
                    requires_confirmation=True,
                    warnings=["紧急呼叫将立即执行"],
                )
            return SafetyCheckResult(is_safe=True)

        # Check if action is blocked while driving
        if driving_state in ("driving", "highway"):
            action_name = intent.intent_type.value
            if action_name in self.blocked_while_driving:
                return SafetyCheckResult(
                    is_safe=False,
                    blocked_reason=f"操作 '{action_name}' 在行驶中被禁止",
                )

            # Check confirmation requirements
            check_key = action_name
            if driving_state == "highway" and intent.intent_type == IntentType.OPEN_WINDOW:
                check_key = "open_window_highway"
            if check_key in self.require_confirmation:
                return SafetyCheckResult(
                    is_safe=True,
                    requires_confirmation=True,
                    warnings=[f"操作 '{action_name}' 需要确认"],
                )

            # Warn about complex operations while driving
            if intent.domain == DomainType.NAVIGATION and driving_state == "highway":
                warnings.append("高速行驶中，建议使用语音交互完成导航设置")

        return SafetyCheckResult(is_safe=True, warnings=warnings)
