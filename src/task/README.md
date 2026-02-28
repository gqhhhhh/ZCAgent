# Task 模块 —— 任务 DAG 调度系统

## 概述

`task/` 实现了基于有向无环图（DAG）的任务管理系统，支持：

- **任务依赖建模** — 自动分析任务间先后关系
- **并行执行** — 无依赖关系的任务可同时执行
- **优先级调度** — 安全任务始终优先，可抢占低优先级任务
- **冲突处理** — 同领域新任务自动取消旧任务（如新导航替代旧导航）

## 文件说明

| 文件 | 作用 |
|------|------|
| `task_graph.py` | 任务 DAG：`Task` 数据结构 + `TaskGraph` 依赖管理 + 执行波次计算 |
| `task_executor.py` | 任务执行器：将 `ParsedIntent` 转为 `Task`，按领域分发到对应处理函数 |
| `task_scheduler.py` | 优先级调度器：并行限制、安全抢占、冲突取消 |

## 技术栈

- **DAG 拓扑排序** — `get_execution_order()` 按依赖关系分组为并行执行波次
- **Enum 状态机** — `TaskStatus` 定义任务生命周期：PENDING → READY → RUNNING → COMPLETED/FAILED/CANCELLED
- **策略模式** — `_handlers` 字典映射动作到处理函数，支持 `register_handler()` 扩展

## 优先级定义

| 优先级 | 领域 | 值 |
|--------|------|-----|
| CRITICAL | 安全（SOS、ADAS） | 100 |
| HIGH | 导航 | 80 |
| — | 电话 | 70 |
| MEDIUM | 车辆设置 | 60 |
| NORMAL | 音乐 | 50 |
| LOW | 通用/后台 | 30 |

## 快速使用

```python
from src.task.task_graph import Task, TaskGraph

graph = TaskGraph()

# 创建任务
t1 = Task(task_id="nav", name="导航", domain="navigation",
          action="navigate_to", priority=80)
t2 = Task(task_id="music", name="播放音乐", domain="music",
          action="play_music", priority=50)

graph.add_task(t1)
graph.add_task(t2)
graph.add_dependency("music", "nav")  # 音乐依赖导航完成

# 获取执行顺序
waves = graph.get_execution_order()
# [["nav"], ["music"]]  — 先导航，再播放音乐
```

## 扩展方式

- 通过 `TaskExecutor.register_handler(action, fn)` 注册自定义任务处理函数。
- 修改 `DOMAIN_PRIORITY` 字典调整各领域默认优先级。
