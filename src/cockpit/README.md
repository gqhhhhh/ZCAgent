# Cockpit 模块 —— 座舱语义理解与安全检查

## 概述

`cockpit/` 是面向汽车智能座舱的语义层，负责：

1. **意图解析**：将自然语言转换为结构化的 `ParsedIntent`（意图类型 + 领域 + 槽位）。
2. **安全检查**：根据驾驶状态（停车/行驶/高速）判断操作是否安全。

## 文件说明

| 文件 | 作用 |
|------|------|
| `domains.py` | 定义领域枚举（`DomainType`）、意图枚举（`IntentType`）和 `ParsedIntent` 数据结构 |
| `intent_parser.py` | 关键词匹配 + LLM 双重意图解析器，含槽位提取（目的地、联系人、温度等） |
| `safety_checker.py` | 基于规则的安全引擎：行驶中阻止危险操作，高速场景要求二次确认 |

## 技术栈

- **Python Enum** — 类型安全的领域 & 意图定义，避免字符串拼写错误
- **正则表达式** — 从文本中提取温度、音量等槽位值
- **关键词权重评分** — 匹配关键词长度占比 × 放大系数计算置信度
- **YAML 配置驱动** — 安全规则从 `config/config.yaml` 加载，无需改代码

## 意图解析流程

```
用户文本 → 关键词匹配（中英文关键词表）
         → 置信度 > 0.5?
            是 → 返回匹配结果 + 槽位提取
            否 → LLM 语义解析（如可用）
         → 返回 ParsedIntent
```

## 安全检查规则

| 场景 | 行为 |
|------|------|
| 停车状态 | 所有操作放行 |
| 行驶中 + 被禁操作 | 直接阻止（如看视频、浏览网页） |
| 高速 + 紧急呼叫 | 放行，但要求确认 |
| 高速 + 开窗 | 要求确认后执行 |
| 安全类操作（SOS） | 始终放行，最高优先级 |

## 快速使用

```python
from src.cockpit.intent_parser import IntentParser
from src.cockpit.safety_checker import SafetyChecker

# 意图解析
parser = IntentParser()
intent = parser.parse("导航到天安门")
print(intent.intent_type)  # IntentType.NAVIGATE_TO
print(intent.slots)        # {"destination": "天安门"}

# 安全检查
checker = SafetyChecker()
result = checker.check(intent, driving_state="driving")
print(result.is_safe)  # True
```

## 扩展方式

- 在 `INTENT_KEYWORDS` 字典中添加新关键词即可支持新意图。
- 在 `config/config.yaml` 的 `safety` 部分修改黑名单和确认列表。
