# Agent 模块 —— 多 Agent 协作核心

## 概述

`agent/` 是 ZCAgent 的决策中枢，负责理解用户意图并调度任务执行。采用 **双路径架构**：

- **快速路径**：高置信度意图直接交给 `PlanExecuteAgent` 执行，响应延迟低。
- **深度路径**：低置信度 / 歧义请求先经 `CoTAgent` 链式推理，再进入任务规划。

## 文件说明

| 文件 | 作用 |
|------|------|
| `base_agent.py` | Agent 抽象基类，定义 `process()` 接口和统一的 `AgentResponse` 数据结构 |
| `cot_agent.py` | Chain-of-Thought 推理 Agent，通过逐步分析处理复杂/多意图请求 |
| `plan_execute_agent.py` | Plan-and-Execute Agent，将意图转为任务 DAG 并按依赖顺序执行 |
| `dispatcher.py` | 中央调度器，串联意图解析 → 安全检查 → CoT/快速路径 → 任务执行 → 记忆管理 |

## 技术栈

- **Python dataclass** — `AgentResponse` 作为类型安全的返回值容器
- **ABC 抽象基类** — 保证所有 Agent 实现统一的 `process()` 接口
- **策略模式** — `dispatcher.py` 根据置信度阈值动态选择推理路径

## 核心流程

```
用户输入 → IntentParser 解析意图
         → SafetyChecker 安全校验
         → 置信度 ≥ 阈值?
            是 → PlanExecuteAgent（快速路径）
            否 → CoTAgent 深度推理 → PlanExecuteAgent
         → MemoryManager 持久化上下文
```

## 快速使用

```python
from src.agent.dispatcher import AgentDispatcher

dispatcher = AgentDispatcher()

# 简单指令（快速路径）
response = dispatcher.process("导航到天安门")
print(response.content)  # "导航到: 天安门"

# 复杂指令（深度路径）
response = dispatcher.process("导航到天安门，顺便放首爵士乐")

# 安全检查
response = dispatcher.process("看视频", driving_state="driving")
print(response.content)  # "操作被阻止: ..."
```

## 扩展方式

1. 继承 `BaseAgent` 并实现 `process()` 方法即可添加新 Agent。
2. 在 `dispatcher.py` 中注册新路径以接入自定义推理策略。
