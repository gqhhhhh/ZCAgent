"""Chain-of-Thought agent for deep semantic understanding.

CoT Agent 通过逐步推理分析用户意图，适用于模糊或多意图请求。
当 LLM 可用时进行链式思维分析，否则降级到基于规则的意图解析。
"""

import json
import logging

from src.agent.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class CoTAgent(BaseAgent):
    """Chain-of-Thought agent for complex reasoning.

    Uses step-by-step reasoning to deeply understand user intent,
    especially for ambiguous or multi-part requests.
    """

    def __init__(self, llm_client=None):
        super().__init__(name="cot_agent", llm_client=llm_client)

    def process(self, user_input: str, context: dict | None = None) -> AgentResponse:
        """Process input using chain-of-thought reasoning.

        Args:
            user_input: User's natural language input.
            context: Memory and conversation context.

        Returns:
            AgentResponse with reasoning results.
        """
        context = context or {}

        if self.llm_client is None:
            return self._rule_based_process(user_input, context)

        return self._llm_process(user_input, context)

    def _build_system_prompt(self) -> str:
        return (
            "你是智能座舱的语义理解Agent。请使用链式思维(Chain-of-Thought)逐步分析用户意图。\n\n"
            "分析步骤：\n"
            "1. 理解用户字面意思\n"
            "2. 分析上下文和隐含意图\n"
            "3. 识别所有需要执行的操作\n"
            "4. 评估安全性和优先级\n"
            "5. 生成最终理解结果\n\n"
            "返回JSON格式：\n"
            '{"reasoning": "推理过程", "intents": [{"type": "意图类型", '
            '"confidence": 0.0-1.0, "slots": {}}], "response": "回复用户的话"}'
        )

    def _llm_process(self, user_input: str, context: dict) -> AgentResponse:
        """Use LLM for chain-of-thought analysis."""
        messages = [{"role": "system", "content": self._build_system_prompt()}]

        # Add context
        if context.get("recent_messages"):
            for msg in context["recent_messages"][-3:]:
                messages.append(msg)

        if context.get("preferences"):
            pref_text = "\n".join(
                f"- {p['key']}: {p['content']}" for p in context["preferences"]
            )
            messages.append({
                "role": "system",
                "content": f"用户偏好：\n{pref_text}",
            })

        messages.append({"role": "user", "content": user_input})

        try:
            result = self.llm_client.generate_json(messages)
            intents = result.get("intents", [])
            return AgentResponse(
                content=result.get("response", ""),
                intent_results=intents,
                confidence=max((i.get("confidence", 0) for i in intents), default=0.0),
                metadata={"reasoning": result.get("reasoning", "")},
            )
        except Exception as e:
            logger.warning("CoT LLM processing failed: %s", e)
            return self._rule_based_process(user_input, context)

    def _rule_based_process(self, user_input: str, context: dict) -> AgentResponse:
        """Fallback rule-based processing when LLM is unavailable."""
        from src.cockpit.intent_parser import IntentParser

        parser = IntentParser()
        intent = parser.parse(user_input)

        return AgentResponse(
            content=f"已识别意图: {intent.intent_type.value}",
            intent_results=[{
                "type": intent.intent_type.value,
                "confidence": intent.confidence,
                "slots": intent.slots,
                "domain": intent.domain.value,
            }],
            confidence=intent.confidence,
        )
