# Tools 模块 —— 外部 API 工具

## 概述

`tools/` 封装了外部 API 调用能力，为 Agent 提供真实世界数据接入。**未配置 API Key 时自动返回模拟数据**，保证开发和测试流程不中断。

## 文件说明

| 文件 | 作用 |
|------|------|
| `base_tool.py` | 工具抽象基类，定义 `run()` 接口和 `ToolResult` 统一返回结构 |
| `amap_tool.py` | 高德地图 API：POI 搜索、地理编码、驾车路线规划 |
| `web_search_tool.py` | 网页搜索 API：基于 Bing Web Search API 的信息检索 |

## 技术栈

- **urllib（标准库）** — HTTP 请求，无需额外依赖
- **模拟降级** — 无 API Key 时返回结构一致的模拟数据（`simulated: true`）
- **JSON Schema** — `get_schema()` 返回工具参数描述，供 LLM function calling 使用

## 环境变量

| 变量 | 说明 |
|------|------|
| `AMAP_API_KEY` | 高德地图 Web 服务 API Key |
| `WEB_SEARCH_API_KEY` | Bing Web Search API Key |
| `WEB_SEARCH_ENDPOINT` | 自定义搜索 API 端点（默认 Bing） |

## 快速使用

```python
from src.tools.amap_tool import AmapTool
from src.tools.web_search_tool import WebSearchTool

# 高德地图 POI 搜索
amap = AmapTool()
result = amap.run(action="poi_search", keywords="加油站", city="北京")
print(result.data)  # {"pois": [...], "count": 1, "simulated": True}

# 地理编码
result = amap.run(action="geocode", address="北京市天安门")
print(result.data)  # {"formatted_address": "...", "location": "..."}

# 驾车路线
result = amap.run(action="route", origin="116.4,39.9", destination="116.5,40.0")
print(result.data)  # {"distance": "12000", "duration": "1800"}

# 网页搜索
search = WebSearchTool()
result = search.run(query="今天北京天气")
print(result.data)  # {"results": [...], "count": 3, "simulated": True}
```

## 扩展方式

1. 继承 `BaseTool` 并实现 `run()` 和 `get_schema()` 方法。
2. 在 `__init__.py` 中注册新工具类以便统一导入。
