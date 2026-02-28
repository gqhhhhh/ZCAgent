"""Plan-and-Execute agent for multi-step task planning and execution.

将用户意图分解为可执行的任务 DAG（有向无环图），按依赖顺序逐波执行。
支持安全检查拦截、任务失败回退和多任务并行处理。
"""

import json
import logging
import uuid

from src.agent.base_agent import BaseAgent, AgentResponse
from src.task.task_graph import Task, TaskGraph
from src.task.task_executor import TaskExecutor, DOMAIN_PRIORITY
from src.cockpit.domains import DomainType, IntentType, ParsedIntent, INTENT_DOMAIN_MAP

logger = logging.getLogger(__name__)


class PlanExecuteAgent(BaseAgent):
    """Plan-and-Execute agent for structured task execution.

    Decomposes complex user requests into a plan of subtasks,
    then executes them in dependency order with safety checks.
    Designed for fast, safe decision-making.
    """

    def __init__(self, llm_client=None, safety_checker=None, executor=None):
        super().__init__(name="plan_execute_agent", llm_client=llm_client)
        self.safety_checker = safety_checker
        self.executor = executor or TaskExecutor()

    def process(self, user_input: str, context: dict | None = None) -> AgentResponse:
        """Process input by planning and executing tasks.

        Args:
            user_input: User's natural language input.
            context: Memory and conversation context.

        Returns:
            AgentResponse with task execution results.
        """
        context = context or {}
        intent_results = context.get("intent_results", [])

        if not intent_results:
            return AgentResponse(
                content="无法识别可执行的任务",
                confidence=0.0,
            )

        # 根据意图列表构建任务 DAG
        graph = TaskGraph()
        tasks = []

        for i, intent_data in enumerate(intent_results):
            # 将字符串意图类型映射回枚举，无法识别时标记为 UNKNOWN
            try:
                intent_type = IntentType(intent_data.get("type", "unknown"))
            except ValueError:
                intent_type = IntentType.UNKNOWN

            domain = INTENT_DOMAIN_MAP.get(intent_type, DomainType.GENERAL)
            intent = ParsedIntent(
                intent_type=intent_type,
                domain=domain,
                confidence=intent_data.get("confidence", 0.5),
                slots=intent_data.get("slots", {}),
                raw_text=user_input,
            )

            task = self.executor.intent_to_task(intent)
            tasks.append(task)

            # Safety check
            if self.safety_checker:
                driving_state = context.get("driving_state", "parked")
                safety_result = self.safety_checker.check(intent, driving_state)
                if not safety_result.is_safe:
                    task.mark_cancelled()
                    task.error = safety_result.blocked_reason

            graph.add_task(task)

        # 建立任务间依赖：非安全任务依赖所有安全任务先完成
        safety_task_ids = [t.task_id for t in tasks if t.domain == "safety"]
        for task in tasks:
            if task.domain != "safety" and safety_task_ids:
                for sid in safety_task_ids:
                    graph.add_dependency(task.task_id, sid)

        # 按 DAG 拓扑排序执行：每一波内的任务可并行执行
        task_results = []
        execution_waves = graph.get_execution_order()

        for wave in execution_waves:
            for task_id in wave:
                task = graph.get_task(task_id)
                if task and task.status.value not in ("cancelled", "failed"):
                    try:
                        result = self.executor.execute(task)
                        graph.complete_task(task_id, result)
                        task_results.append(result)
                    except Exception as e:
                        graph.fail_task(task_id, str(e))
                        task_results.append({
                            "status": "failed",
                            "task": task_id,
                            "error": str(e),
                        })

        # 组装最终响应：根据任务成功/失败比例生成用户可读的结果文本
        success_count = sum(1 for r in task_results if r.get("status") == "success")
        total = len(task_results)

        if total == 0:
            content = "所有任务已被取消或无法执行"
        elif success_count == total:
            messages = [r.get("message", "") for r in task_results if r.get("message")]
            content = "；".join(messages) if messages else "所有任务执行完成"
        else:
            content = f"已完成 {success_count}/{total} 个任务"

        requires_confirmation = any(
            t.error and "需要确认" in t.error for t in tasks
        )

        return AgentResponse(
            content=content,
            task_results=task_results,
            requires_confirmation=requires_confirmation,
            confidence=success_count / max(total, 1),
            metadata={"execution_waves": execution_waves},
        )
