"""Domain definitions for the intelligent cockpit system.

定义座舱系统的领域类型（导航/电话/音乐/车控/安全/通用）和意图类型枚举，
以及意图到领域的映射关系。所有模块共享这些类型定义以保证一致性。
"""

from dataclasses import dataclass, field
from enum import Enum


class DomainType(Enum):
    """Supported cockpit function domains."""
    NAVIGATION = "navigation"
    PHONE = "phone"
    MUSIC = "music"
    VEHICLE_SETTING = "vehicle_setting"
    SAFETY = "safety"
    GENERAL = "general"


class IntentType(Enum):
    """Common intent types across domains."""
    # Navigation
    NAVIGATE_TO = "navigate_to"
    SEARCH_POI = "search_poi"
    CANCEL_NAVIGATION = "cancel_navigation"
    # Phone
    MAKE_CALL = "make_call"
    ANSWER_CALL = "answer_call"
    REJECT_CALL = "reject_call"
    SEND_MESSAGE = "send_message"
    # Music
    PLAY_MUSIC = "play_music"
    PAUSE_MUSIC = "pause_music"
    NEXT_TRACK = "next_track"
    ADJUST_VOLUME = "adjust_volume"
    # Vehicle settings
    SET_TEMPERATURE = "set_temperature"
    OPEN_WINDOW = "open_window"
    CLOSE_WINDOW = "close_window"
    SET_SEAT = "set_seat"
    SET_LIGHT = "set_light"
    # Safety
    EMERGENCY_CALL = "emergency_call"
    ADAS_CONTROL = "adas_control"
    FATIGUE_ALERT = "fatigue_alert"
    # General
    QUERY = "query"
    CHAT = "chat"
    UNKNOWN = "unknown"


# Mapping from intent to domain
INTENT_DOMAIN_MAP: dict[IntentType, DomainType] = {
    IntentType.NAVIGATE_TO: DomainType.NAVIGATION,
    IntentType.SEARCH_POI: DomainType.NAVIGATION,
    IntentType.CANCEL_NAVIGATION: DomainType.NAVIGATION,
    IntentType.MAKE_CALL: DomainType.PHONE,
    IntentType.ANSWER_CALL: DomainType.PHONE,
    IntentType.REJECT_CALL: DomainType.PHONE,
    IntentType.SEND_MESSAGE: DomainType.PHONE,
    IntentType.PLAY_MUSIC: DomainType.MUSIC,
    IntentType.PAUSE_MUSIC: DomainType.MUSIC,
    IntentType.NEXT_TRACK: DomainType.MUSIC,
    IntentType.ADJUST_VOLUME: DomainType.MUSIC,
    IntentType.SET_TEMPERATURE: DomainType.VEHICLE_SETTING,
    IntentType.OPEN_WINDOW: DomainType.VEHICLE_SETTING,
    IntentType.CLOSE_WINDOW: DomainType.VEHICLE_SETTING,
    IntentType.SET_SEAT: DomainType.VEHICLE_SETTING,
    IntentType.SET_LIGHT: DomainType.VEHICLE_SETTING,
    IntentType.EMERGENCY_CALL: DomainType.SAFETY,
    IntentType.ADAS_CONTROL: DomainType.SAFETY,
    IntentType.FATIGUE_ALERT: DomainType.SAFETY,
    IntentType.QUERY: DomainType.GENERAL,
    IntentType.CHAT: DomainType.GENERAL,
    IntentType.UNKNOWN: DomainType.GENERAL,
}


@dataclass
class ParsedIntent:
    """Represents a parsed user intent."""
    intent_type: IntentType
    domain: DomainType | None = None
    confidence: float = 0.0
    slots: dict = field(default_factory=dict)
    raw_text: str = ""

    def __post_init__(self):
        if self.domain is None:
            self.domain = INTENT_DOMAIN_MAP.get(self.intent_type, DomainType.GENERAL)
