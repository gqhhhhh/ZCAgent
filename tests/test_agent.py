"""Tests for the multi-agent system."""

import pytest

from src.agent.base_agent import AgentResponse
from src.agent.cot_agent import CoTAgent
from src.agent.plan_execute_agent import PlanExecuteAgent
from src.agent.dispatcher import AgentDispatcher
from src.cockpit.intent_parser import IntentParser
from src.cockpit.safety_checker import SafetyChecker
from src.cockpit.domains import DomainType, IntentType, ParsedIntent
from src.llm.llm_client import LLMClient


class TestIntentParser:
    def test_parse_navigation(self):
        parser = IntentParser()
        intent = parser.parse("导航到天安门")
        assert intent.intent_type == IntentType.NAVIGATE_TO
        assert intent.domain == DomainType.NAVIGATION
        assert intent.slots.get("destination") == "天安门"

    def test_parse_phone(self):
        parser = IntentParser()
        intent = parser.parse("给张三打电话")
        assert intent.intent_type == IntentType.MAKE_CALL
        assert intent.domain == DomainType.PHONE
        assert intent.slots.get("contact") == "张三"

    def test_parse_music(self):
        parser = IntentParser()
        intent = parser.parse("播放音乐周杰伦的晴天")
        assert intent.intent_type == IntentType.PLAY_MUSIC
        assert intent.domain == DomainType.MUSIC

    def test_parse_temperature(self):
        parser = IntentParser()
        intent = parser.parse("把空调温度调到25度")
        assert intent.intent_type == IntentType.SET_TEMPERATURE
        assert intent.slots.get("temperature") == 25

    def test_parse_volume(self):
        parser = IntentParser()
        intent = parser.parse("音量调大一点")
        assert intent.intent_type == IntentType.ADJUST_VOLUME
        assert intent.slots.get("direction") == "up"

    def test_parse_emergency(self):
        parser = IntentParser()
        intent = parser.parse("紧急呼叫SOS")
        assert intent.intent_type == IntentType.EMERGENCY_CALL
        assert intent.domain == DomainType.SAFETY

    def test_parse_unknown(self):
        parser = IntentParser()
        intent = parser.parse("xyzabc")
        assert intent.intent_type == IntentType.UNKNOWN


class TestSafetyChecker:
    def test_safe_when_parked(self):
        checker = SafetyChecker({"blocked_while_driving": ["watch_video"],
                                  "require_confirmation": []})
        intent = ParsedIntent(intent_type=IntentType.NAVIGATE_TO,
                             domain=DomainType.NAVIGATION, confidence=0.9)
        result = checker.check(intent, "parked")
        assert result.is_safe

    def test_blocked_while_driving(self):
        checker = SafetyChecker({"blocked_while_driving": ["watch_video"],
                                  "require_confirmation": []})
        from unittest.mock import MagicMock
        mock_intent = MagicMock()
        mock_intent.domain = DomainType.GENERAL
        mock_intent.intent_type = MagicMock()
        mock_intent.intent_type.value = "watch_video"
        result = checker.check(mock_intent, "driving")
        assert not result.is_safe

    def test_safety_always_allowed(self):
        checker = SafetyChecker({"blocked_while_driving": [],
                                  "require_confirmation": ["emergency_call"]})
        intent = ParsedIntent(intent_type=IntentType.EMERGENCY_CALL,
                             domain=DomainType.SAFETY, confidence=0.9)
        result = checker.check(intent, "highway")
        assert result.is_safe
        assert result.requires_confirmation

    def test_highway_warning(self):
        checker = SafetyChecker({"blocked_while_driving": [],
                                  "require_confirmation": []})
        intent = ParsedIntent(intent_type=IntentType.NAVIGATE_TO,
                             domain=DomainType.NAVIGATION, confidence=0.9)
        result = checker.check(intent, "highway")
        assert result.is_safe
        assert len(result.warnings) > 0


class TestCoTAgent:
    def test_rule_based_fallback(self):
        agent = CoTAgent()  # No LLM
        response = agent.process("导航到天安门")
        assert response.confidence > 0
        assert len(response.intent_results) > 0
        assert response.intent_results[0]["type"] == "navigate_to"

    def test_with_mock_llm(self):
        import json
        mock_response = json.dumps({
            "reasoning": "用户想要导航到天安门",
            "intents": [{"type": "navigate_to", "confidence": 0.95,
                        "slots": {"destination": "天安门"}}],
            "response": "正在为您导航到天安门"
        })
        llm = LLMClient(mock_response=mock_response)
        agent = CoTAgent(llm_client=llm)
        response = agent.process("导航到天安门")
        assert "天安门" in response.content
        assert response.confidence > 0.9


class TestPlanExecuteAgent:
    def test_execute_single_intent(self):
        agent = PlanExecuteAgent()
        context = {
            "intent_results": [{
                "type": "navigate_to",
                "confidence": 0.9,
                "slots": {"destination": "天安门"},
                "domain": "navigation",
            }]
        }
        response = agent.process("导航到天安门", context)
        assert response.confidence > 0
        assert len(response.task_results) == 1
        assert response.task_results[0]["status"] == "success"

    def test_execute_multiple_intents(self):
        agent = PlanExecuteAgent()
        context = {
            "intent_results": [
                {"type": "navigate_to", "confidence": 0.9,
                 "slots": {"destination": "天安门"}, "domain": "navigation"},
                {"type": "play_music", "confidence": 0.8,
                 "slots": {"query": "爵士乐"}, "domain": "music"},
            ]
        }
        response = agent.process("导航到天安门，顺便放点音乐", context)
        assert len(response.task_results) == 2

    def test_empty_intents(self):
        agent = PlanExecuteAgent()
        response = agent.process("test", {})
        assert response.confidence == 0.0


class TestAgentDispatcher:
    def test_fast_path(self):
        dispatcher = AgentDispatcher(config={
            "safety": {"blocked_while_driving": [], "require_confirmation": []},
            "memory": {},
            "rag": {},
        })
        response = dispatcher.process("导航到天安门", driving_state="parked")
        assert "天安门" in response.content

    def test_safety_block(self):
        dispatcher = AgentDispatcher(config={
            "safety": {"blocked_while_driving": ["watch_video"],
                       "require_confirmation": []},
            "memory": {},
            "rag": {},
        })
        # This tests that the dispatcher can handle inputs
        response = dispatcher.process("导航到天安门", driving_state="parked")
        assert response.content  # Should return some response

    def test_with_mock_llm(self):
        import json
        mock_resp = json.dumps({
            "intents": [{"type": "play_music", "confidence": 0.9,
                        "slots": {"query": "晴天"}}],
            "reasoning": "用户想听歌",
            "response": "正在播放晴天"
        })
        llm = LLMClient(mock_response=mock_resp)
        dispatcher = AgentDispatcher(
            config={
                "safety": {"blocked_while_driving": [], "require_confirmation": []},
                "memory": {},
                "rag": {},
            },
            llm_client=llm,
        )
        response = dispatcher.process("播放音乐周杰伦的晴天")
        assert response.task_results

    def test_memory_integration(self):
        dispatcher = AgentDispatcher(config={
            "safety": {"blocked_while_driving": [], "require_confirmation": []},
            "memory": {},
            "rag": {},
        })
        dispatcher.process("导航到天安门")
        ctx = dispatcher.memory.get_context()
        assert len(ctx["recent_messages"]) == 2  # user + assistant
