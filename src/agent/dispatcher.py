"""Agent dispatcher coordinating the multi-agent system."""

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

        # Confidence threshold for fast vs deep path
        self._fast_path_threshold = 0.6

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

        # Step 1: Parse intent
        intent = self.intent_parser.parse(user_input)
        logger.info("Parsed intent: %s (confidence=%.2f)",
                    intent.intent_type.value, intent.confidence)

        # Step 2: Safety check
        safety_result = self.safety_checker.check(intent, driving_state)
        if not safety_result.is_safe:
            response = AgentResponse(
                content=f"操作被阻止: {safety_result.blocked_reason}",
                confidence=1.0,
                metadata={"safety_blocked": True},
            )
            self.memory.add_assistant_message(response.content)
            return response

        # Step 3: Choose fast path or deep path
        context = self.memory.get_context()
        context["driving_state"] = driving_state

        if intent.confidence >= self._fast_path_threshold:
            # Fast path: direct to plan-execute
            intent_results = [{
                "type": intent.intent_type.value,
                "confidence": intent.confidence,
                "slots": intent.slots,
                "domain": intent.domain.value,
            }]
            context["intent_results"] = intent_results
            response = self.plan_agent.process(user_input, context)
        else:
            # Deep path: CoT reasoning first
            cot_response = self.cot_agent.process(user_input, context)

            if cot_response.intent_results:
                context["intent_results"] = cot_response.intent_results
                response = self.plan_agent.process(user_input, context)
                response.metadata["cot_reasoning"] = cot_response.metadata.get(
                    "reasoning", ""
                )
            else:
                response = cot_response

        # Handle confirmation requirement
        if safety_result.requires_confirmation:
            response.requires_confirmation = True
            response.content = f"[需要确认] {response.content}"

        # Record response in memory
        self.memory.add_assistant_message(response.content)

        return response
