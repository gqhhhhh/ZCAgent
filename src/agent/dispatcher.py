"""Agent dispatcher coordinating the multi-agent system.

中央调度器：串联意图解析、安全检查、推理路径选择、任务执行和记忆管理。
采用双路径策略——高置信度走快速路径直接执行，低置信度走 CoT 深度推理。
"""

import logging
import os

import yaml

from src.agent.base_agent import AgentResponse
from src.agent.cot_agent import CoTAgent
from src.agent.plan_execute_agent import PlanExecuteAgent
from src.cockpit.intent_parser import IntentParser
from src.cockpit.safety_checker import SafetyChecker
from src.memory.memory_manager import MemoryManager
from src.rag.hybrid_retriever import HybridRetriever
from src.task.task_executor import TaskExecutor
from src.task.task_scheduler import TaskScheduler

logger = logging.getLogger(__name__)

# 快速路径置信度阈值：意图置信度 ≥ 此值时直接进入 PlanExecute，跳过 CoT 推理
DEFAULT_FAST_PATH_THRESHOLD = 0.6


class AgentDispatcher:
    """Central dispatcher coordinating the multi-agent system.

    Orchestrates the flow between:
    1. Intent parsing (keyword + LLM)
    2. CoT reasoning (deep understanding)
    3. RAG retrieval (knowledge augmentation)
    4. Plan-and-Execute (task execution)
    5. Memory management (context tracking)

    Implements a two-stage approach:
    - Fast path: Simple intents go directly to Plan-Execute
    - Deep path: Complex/ambiguous intents go through CoT first
    """

    def __init__(self, config: dict | None = None, llm_client=None):
        if config is None:
            config = self._load_config()

        self.llm_client = llm_client
        self.intent_parser = IntentParser(llm_client=llm_client)
        self.safety_checker = SafetyChecker(config.get("safety", {}))
        self.memory = MemoryManager(
            config=config.get("memory", {}), llm_client=llm_client
        )
        self.retriever = HybridRetriever(config.get("rag", {}))
        self.executor = TaskExecutor()
        self.scheduler = TaskScheduler()

        self.cot_agent = CoTAgent(llm_client=llm_client)
        self.plan_agent = PlanExecuteAgent(
            llm_client=llm_client,
            safety_checker=self.safety_checker,
            executor=self.executor,
        )

        # 快速路径置信度阈值（可通过 config 覆盖）
        self._fast_path_threshold = config.get(
            "fast_path_threshold", DEFAULT_FAST_PATH_THRESHOLD
        )

    def _load_config(self) -> dict:
        """Load configuration from config.yaml."""
        config_paths = [
            os.path.join(os.path.dirname(__file__), "../../config/config.yaml"),
            "config/config.yaml",
        ]
        for path in config_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        return {}

    def process(self, user_input: str,
                driving_state: str = "parked") -> AgentResponse:
        """Process a user input through the full agent pipeline.

        Args:
            user_input: User's natural language input.
            driving_state: Current driving state ('parked', 'driving', 'highway').

        Returns:
            AgentResponse with the final result.
        """
        # Record in memory
        self.memory.add_user_message(user_input)

        # Step 1: 意图解析 — 关键词匹配 + LLM 双重策略
        intent = self.intent_parser.parse(user_input)
        logger.info("Parsed intent: %s (confidence=%.2f)",
                    intent.intent_type.value, intent.confidence)

        # Step 2: 安全检查 — 行驶中阻止危险操作
        safety_result = self.safety_checker.check(intent, driving_state)
        if not safety_result.is_safe:
            response = AgentResponse(
                content=f"操作被阻止: {safety_result.blocked_reason}",
                confidence=1.0,
                metadata={"safety_blocked": True},
            )
            self.memory.add_assistant_message(response.content)
            return response

        # Step 3: 根据置信度选择快速路径或深度路径
        context = self.memory.get_context()
        context["driving_state"] = driving_state

        if intent.confidence >= self._fast_path_threshold:
            # 快速路径：置信度足够高，直接构建任务执行
            intent_results = [{
                "type": intent.intent_type.value,
                "confidence": intent.confidence,
                "slots": intent.slots,
                "domain": intent.domain.value,
            }]
            context["intent_results"] = intent_results
            response = self.plan_agent.process(user_input, context)
        else:
            # 深度路径：置信度不足，先通过 CoT 链式推理分析
            cot_response = self.cot_agent.process(user_input, context)

            if cot_response.intent_results:
                context["intent_results"] = cot_response.intent_results
                response = self.plan_agent.process(user_input, context)
                response.metadata["cot_reasoning"] = cot_response.metadata.get(
                    "reasoning", ""
                )
            else:
                response = cot_response

        # 需要用户确认的操作（如紧急呼叫、高速开窗）添加确认标记
        if safety_result.requires_confirmation:
            response.requires_confirmation = True
            response.content = f"[需要确认] {response.content}"

        # 将本次交互记录到记忆系统，供后续对话参考
        self.memory.add_assistant_message(response.content)

        return response
