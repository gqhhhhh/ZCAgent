# Memory 模块 —— 三层记忆系统

## 概述

`memory/` 实现了仿人类的三层记忆架构，为对话系统提供上下文保持和用户偏好学习能力：

| 层级 | 类比 | 特点 |
|------|------|------|
| **工作记忆** | 大脑工作台 | 容量有限，保存当前对话最相关的信息；超容量时淘汰最不重要的条目 |
| **短期记忆** | 短期记忆 | 保存最近的对话历史，带 TTL 自动过期机制 |
| **长期记忆** | 长期记忆 | 持久化用户偏好和事实，支持冲突消解和 LLM 摘要 |

## 文件说明

| 文件 | 作用 |
|------|------|
| `working_memory.py` | 工作记忆：基于 `OrderedDict` 的 LRU + 重要性淘汰 |
| `short_term_memory.py` | 短期记忆：带 TTL 过期的对话历史队列 |
| `long_term_memory.py` | 长期记忆：偏好/事实存储，冲突消解 + LLM 摘要提炼 |
| `memory_manager.py` | 统一管理器：协调三层记忆的读写和上下文组装 |

## 技术栈

- **OrderedDict** — 工作记忆的插入顺序维护 + O(1) 查找
- **TTL 过期** — 短期记忆按时间戳自动清理过期条目
- **冲突消解策略** — 新旧信息冲突时，高重要性覆盖低重要性，否则记录冲突历史
- **LLM 摘要** — 当待处理条目积累到阈值时，调用 LLM 提取重要偏好存入长期记忆

## 快速使用

```python
from src.memory.memory_manager import MemoryManager

mm = MemoryManager()

# 记录对话
mm.add_user_message("导航到天安门")
mm.add_assistant_message("正在为您导航到天安门")

# 存储用户偏好
mm.store_preference("music_genre", "用户喜欢爵士乐")

# 获取统一上下文（传给 Agent）
context = mm.get_context()
# {'working_memory': '...', 'recent_messages': [...], 'preferences': [...]}

# 跨层搜索
results = mm.search("天安门")
```

## 配置

```yaml
memory:
  working_memory_capacity: 10          # 工作记忆最大条目数
  short_term_ttl_seconds: 300          # 短期记忆过期时间（秒）
  long_term_importance_threshold: 0.7  # 长期记忆存储门槛
  summary_trigger_count: 20            # 触发 LLM 摘要的待处理条目数
```
