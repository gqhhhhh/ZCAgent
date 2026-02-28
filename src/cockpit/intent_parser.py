"""Intent parser for cockpit voice commands.

意图解析器：通过关键词匹配 + LLM 双重策略，将自然语言转为结构化的 ParsedIntent。
关键词匹配速度快、确定性强，适合常见指令；LLM 解析处理模糊/复杂表述。
"""

import json
import logging
import re

from src.cockpit.domains import DomainType, IntentType, ParsedIntent, INTENT_DOMAIN_MAP

logger = logging.getLogger(__name__)

# Keyword-based intent detection rules (Chinese + English)
INTENT_KEYWORDS: dict[IntentType, list[str]] = {
    IntentType.NAVIGATE_TO: ["导航到", "导航去", "去往", "navigate to", "go to", "带我去"],
    IntentType.SEARCH_POI: ["搜索", "查找", "附近的", "找一下", "search", "find nearby"],
    IntentType.CANCEL_NAVIGATION: ["取消导航", "停止导航", "cancel navigation"],
    IntentType.MAKE_CALL: ["打电话", "呼叫", "拨打", "call", "dial"],
    IntentType.ANSWER_CALL: ["接电话", "接听", "answer"],
    IntentType.REJECT_CALL: ["拒接", "挂断", "reject", "hang up"],
    IntentType.SEND_MESSAGE: ["发消息", "发短信", "发送", "send message", "text"],
    IntentType.PLAY_MUSIC: ["播放音乐", "播放歌曲", "放一首", "play music", "play song", "来一首"],
    IntentType.PAUSE_MUSIC: ["暂停音乐", "暂停播放", "停止播放", "pause music", "stop music"],
    IntentType.NEXT_TRACK: ["下一首", "切歌", "换一首", "next song", "next track", "skip"],
    IntentType.ADJUST_VOLUME: ["音量", "声音大", "声音小", "volume", "调高音量", "调低音量"],
    IntentType.SET_TEMPERATURE: ["温度", "空调", "制冷", "制热", "temperature", "AC", "暖风"],
    IntentType.OPEN_WINDOW: ["开窗", "打开车窗", "open window"],
    IntentType.CLOSE_WINDOW: ["关窗", "关闭车窗", "close window"],
    IntentType.SET_SEAT: ["座椅", "seat", "座位加热", "座位通风"],
    IntentType.SET_LIGHT: ["车灯", "大灯", "氛围灯", "light", "灯光"],
    IntentType.EMERGENCY_CALL: ["紧急呼叫", "SOS", "emergency", "报警", "急救"],
    IntentType.ADAS_CONTROL: ["自动驾驶", "辅助驾驶", "ADAS", "autopilot", "车道保持"],
    IntentType.FATIGUE_ALERT: ["疲劳", "困了", "fatigue", "tired", "休息提醒"],
}


# Multiplier to scale raw keyword match ratio into confidence score
# Multiplier to scale keyword match ratio into a confidence score.
# A ratio of ~0.33 (keyword length / text length) maps to confidence 1.0.
KEYWORD_MATCH_CONFIDENCE_MULTIPLIER = 3.0


# 关键词匹配置信度阈值：低于此值时尝试 LLM 解析
KEYWORD_CONFIDENCE_THRESHOLD = 0.5


class IntentParser:
    """Parse user input into structured intents using keyword matching and LLM."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def parse(self, text: str) -> ParsedIntent:
        """Parse user text into a structured intent.

        Uses keyword matching first, falls back to LLM-based parsing if available.

        Args:
            text: Raw user input text.

        Returns:
            ParsedIntent with detected intent type, domain, and slots.
        """
        # Try keyword-based matching first
        intent = self._keyword_match(text)
        # 关键词匹配置信度足够高时直接返回，无需 LLM
        if intent.confidence > KEYWORD_CONFIDENCE_THRESHOLD:
            return intent

        # Fall back to LLM-based parsing if available
        if self.llm_client is not None:
            return self._llm_parse(text)

        return ParsedIntent(
            intent_type=IntentType.UNKNOWN,
            domain=DomainType.GENERAL,
            confidence=0.0,
            raw_text=text,
        )

    def _keyword_match(self, text: str) -> ParsedIntent:
        """Match intent using keyword rules."""
        text_lower = text.lower()
        best_intent = IntentType.UNKNOWN
        best_score = 0.0
        matched_keywords = []

        for intent_type, keywords in INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score = len(keyword) / max(len(text), 1)
                    if score > best_score:
                        best_score = score
                        best_intent = intent_type
                        matched_keywords = [keyword]

        confidence = min(best_score * KEYWORD_MATCH_CONFIDENCE_MULTIPLIER, 1.0) if best_score > 0 else 0.0
        domain = INTENT_DOMAIN_MAP.get(best_intent, DomainType.GENERAL)
        slots = self._extract_slots(text, best_intent)

        return ParsedIntent(
            intent_type=best_intent,
            domain=domain,
            confidence=confidence,
            slots=slots,
            raw_text=text,
        )

    def _extract_slots(self, text: str, intent_type: IntentType) -> dict:
        """Extract slot values from text based on intent type."""
        slots = {}

        if intent_type in (IntentType.NAVIGATE_TO, IntentType.SEARCH_POI):
            # Extract destination - text after navigation keywords
            for kw in INTENT_KEYWORDS.get(intent_type, []):
                if kw in text:
                    dest = text.split(kw)[-1].strip()
                    if dest:
                        slots["destination"] = dest
                    break

        elif intent_type in (IntentType.MAKE_CALL, IntentType.SEND_MESSAGE):
            # Extract contact name - look for "给XXX" pattern
            match = re.search(r"给(.+?)(?:打电话|发消息|发短信|$)", text)
            if match:
                slots["contact"] = match.group(1).strip()

        elif intent_type == IntentType.PLAY_MUSIC:
            # Extract song/artist name
            for kw in INTENT_KEYWORDS[IntentType.PLAY_MUSIC]:
                if kw in text:
                    song = text.split(kw)[-1].strip()
                    if song:
                        slots["query"] = song
                    break

        elif intent_type == IntentType.SET_TEMPERATURE:
            # Extract temperature value
            match = re.search(r"(\d+)\s*[度℃°]", text)
            if match:
                slots["temperature"] = int(match.group(1))

        elif intent_type == IntentType.ADJUST_VOLUME:
            match = re.search(r"(\d+)", text)
            if match:
                slots["level"] = int(match.group(1))
            elif any(w in text for w in ["大", "高", "up", "louder"]):
                slots["direction"] = "up"
            elif any(w in text for w in ["小", "低", "down", "quieter"]):
                slots["direction"] = "down"

        return slots

    def _llm_parse(self, text: str) -> ParsedIntent:
        """Use LLM to parse intent when keyword matching fails."""
        intent_list = [it.value for it in IntentType]
        messages = [
            {
                "role": "system",
                "content": (
                    "你是智能座舱语义理解系统。解析用户指令并返回JSON格式：\n"
                    f'{{"intent": "<one of {intent_list}>", '
                    '"confidence": <0.0-1.0>, "slots": {<key-value pairs>}}'
                ),
            },
            {"role": "user", "content": text},
        ]
        try:
            result = self.llm_client.generate_json(messages)
            intent_str = result.get("intent", "unknown")
            try:
                intent_type = IntentType(intent_str)
            except ValueError:
                intent_type = IntentType.UNKNOWN
            domain = INTENT_DOMAIN_MAP.get(intent_type, DomainType.GENERAL)
            return ParsedIntent(
                intent_type=intent_type,
                domain=domain,
                confidence=result.get("confidence", 0.5),
                slots=result.get("slots", {}),
                raw_text=text,
            )
        except Exception as e:
            logger.warning("LLM intent parsing failed: %s", e)
            return ParsedIntent(
                intent_type=IntentType.UNKNOWN,
                domain=DomainType.GENERAL,
                confidence=0.0,
                raw_text=text,
            )
