# LLM 模块 —— 大语言模型抽象层

## 概述

`llm/` 提供统一的 LLM API 调用接口，屏蔽底层 API 差异。内置 **Mock 模式**，开发和测试时无需真实 API Key。

## 文件说明

| 文件 | 作用 |
|------|------|
| `llm_client.py` | LLM 客户端，支持 OpenAI 兼容 API、自定义端点和 Mock 响应 |

## 技术栈

- **OpenAI Python SDK** — 调用 GPT-4 等大语言模型
- **惰性初始化** — `_get_client()` 延迟创建 OpenAI 客户端，未调用时不触发网络请求
- **JSON 提取** — 自动处理 LLM 返回的 Markdown 代码块，提取纯 JSON
- **YAML 配置** — 模型名称、温度、Token 上限均从 `config/config.yaml` 读取

## 配置

通过 `config/config.yaml` 或环境变量配置：

```yaml
llm:
  model: "gpt-4"
  temperature: 0.3
  max_tokens: 2048
  api_base: ""       # 自定义端点（如本地部署的兼容 API）
  api_key: ""        # 或通过 OPENAI_API_KEY 环境变量
```

| 环境变量 | 说明 |
|----------|------|
| `OPENAI_API_KEY` | OpenAI API 密钥 |
| `OPENAI_API_BASE` | 自定义 API 端点 URL |

## 快速使用

```python
from src.llm.llm_client import LLMClient

# 生产模式（需要 API Key）
client = LLMClient()
reply = client.generate([{"role": "user", "content": "你好"}])

# Mock 模式（开发/测试）
client = LLMClient(mock_response='{"intent": "navigate_to"}')
result = client.generate_json([{"role": "user", "content": "导航"}])
print(result)  # {"intent": "navigate_to"}
```

## 设计说明

- `generate()` 返回纯文本，`generate_json()` 在其基础上自动解析 JSON。
- 当 `openai` 包未安装时自动降级为 Mock 模式，保证系统可运行。
