"""ZCAgent × LangChain 多 Agent 协同演示
===========================================

本脚本展示如何用 **真实的 LangChain 工具库** 调用 ZCAgent 能力，以及多个
Agent 如何协同完成一项复杂的座舱任务（导航 + 天气查询 + 音乐播放）。

运行前置条件
-----------
1. 安装依赖::

       pip install langchain langchain-openai langchain-core>=0.3.81

2. 配置环境变量::

       export OPENAI_API_KEY="sk-..."          # OpenAI 或兼容 API Key
       export OPENAI_BASE_URL="..."            # 可选，自定义 API 端点
       export AMAP_API_KEY="..."               # 可选，高德地图 API Key
       export WEB_SEARCH_API_KEY="..."         # 可选，Bing 搜索 API Key

   未配置地图/搜索 API Key 时会使用**模拟数据**，同样可以运行演示。

运行方式
--------
::

    python examples/langchain_multi_agent_demo.py

多 Agent 协同架构
-----------------
本演示实现了以下三层 Agent 协作模式：

::

    ┌──────────────────────────────────────────┐
    │          Supervisor Agent（协调器）        │
    │  接收用户请求，将任务分派给下属专业 Agent   │
    └────────────┬───────────────┬─────────────┘
                 │               │
        ┌────────▼──────┐  ┌─────▼────────────┐
        │ Cockpit Agent │  │  Research Agent  │
        │ 座舱控制专家   │  │  信息检索专家    │
        │ (导航/音乐)   │  │  (天气/搜索)     │
        └───────────────┘  └──────────────────┘

每个 Agent 都有专属工具集，Supervisor 负责路由和结果聚合。
"""

from __future__ import annotations

import os
import sys
import json
import textwrap
from typing import Any

# ---------------------------------------------------------------------------
# 检查依赖
# ---------------------------------------------------------------------------

def _check_deps() -> bool:
    missing = []
    try:
        import langchain  # noqa: F401
    except ImportError:
        missing.append("langchain")
    try:
        import langchain_core  # noqa: F401
    except ImportError:
        missing.append("langchain-core")

    if missing:
        print("❌ 缺少依赖，请先安装：")
        print(f"   pip install {' '.join(missing)} langchain-openai")
        return False
    return True


# ---------------------------------------------------------------------------
# 工具定义（使用 LangChain @tool 装饰器）
# ---------------------------------------------------------------------------

def _build_tools():
    """Build LangChain tools using the @tool decorator pattern."""
    from langchain_core.tools import tool  # type: ignore[import]

    # 添加项目根目录到 sys.path，保证能导入 src 包
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.agent.dispatcher import AgentDispatcher
    from src.tools.amap_tool import AmapTool
    from src.tools.web_search_tool import WebSearchTool

    _dispatcher = AgentDispatcher()
    _amap = AmapTool()
    _search = WebSearchTool()

    @tool
    def cockpit_command(command: str) -> str:
        """执行智能座舱指令，包括导航、音乐播放、电话、车辆控制等。
        输入自然语言指令，如"导航到天安门"、"播放爵士乐"。"""
        response = _dispatcher.process(command)
        return response.content

    @tool
    def map_poi_search(keywords: str, city: str = "") -> str:
        """用高德地图搜索兴趣点（POI），如加油站、餐厅、停车场等。
        参数 keywords 为搜索关键词，city 可选（如"北京"）。"""
        result = _amap.run(action="poi_search", keywords=keywords, city=city)
        return json.dumps(result.data, ensure_ascii=False) if result.success else result.to_text()

    @tool
    def web_search(query: str) -> str:
        """搜索互联网获取最新信息，如天气、新闻、实时路况等。
        输入搜索关键词字符串。"""
        result = _search.run(query=query)
        return json.dumps(result.data, ensure_ascii=False) if result.success else result.to_text()

    return cockpit_command, map_poi_search, web_search


# ---------------------------------------------------------------------------
# 方案一：单 Agent + 多工具（最简单）
# ---------------------------------------------------------------------------

def demo_single_agent(llm: Any) -> None:
    """单个 ReAct Agent 使用所有工具完成复合任务。"""
    print("\n" + "=" * 60)
    print("🤖 方案一：单 Agent + 多工具 (ReAct)")
    print("=" * 60)

    from src.integrations.langchain_adapter import create_react_agent_executor

    executor = create_react_agent_executor(llm=llm, verbose=True)

    query = "导航到最近的加油站，同时查一下北京今天的天气，最后放一首轻音乐"
    print(f"\n用户请求: {query}\n")
    try:
        result = executor.invoke({"input": query})
        print(f"\n✅ 最终回答: {result['output']}")
    except Exception as e:
        print(f"⚠️  执行出错（通常是因为未配置 OPENAI_API_KEY）: {e}")
        # 降级演示：直接调用工具
        _demo_tools_directly()


def _demo_tools_directly() -> None:
    """当 LLM 不可用时，直接调用工具展示功能。"""
    print("\n📡 直接工具调用演示（无需 LLM）:")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.agent.dispatcher import AgentDispatcher
    from src.tools.amap_tool import AmapTool
    from src.tools.web_search_tool import WebSearchTool

    dispatcher = AgentDispatcher()
    amap = AmapTool()
    search = WebSearchTool()

    print("\n  [1] 座舱指令 → 导航到天安门")
    r = dispatcher.process("导航到天安门")
    print(f"      结果: {r.content}")

    print("\n  [2] 高德地图 → 搜索附近加油站")
    r2 = amap.run(action="poi_search", keywords="加油站", city="北京")
    print(f"      结果: {json.dumps(r2.data, ensure_ascii=False, indent=2)[:200]}...")

    print("\n  [3] 网络搜索 → 北京今日天气")
    r3 = search.run(query="北京今天天气")
    print(f"      结果: {json.dumps(r3.data, ensure_ascii=False, indent=2)[:200]}...")

    print("\n  [4] 座舱指令 → 播放轻音乐")
    r4 = dispatcher.process("播放轻音乐")
    print(f"      结果: {r4.content}")


# ---------------------------------------------------------------------------
# 方案二：多 Agent 协同（Supervisor 模式）
# ---------------------------------------------------------------------------

def demo_multi_agent(llm: Any) -> None:
    """用 LangChain LCEL（LangChain Expression Language）实现 Supervisor 模式。

    Supervisor Agent 接收用户请求后判断需要哪些子 Agent，然后并行/串行调用
    专业 Agent，最后汇总结果返回给用户。
    """
    print("\n" + "=" * 60)
    print("🤝 方案二：多 Agent 协同 (Supervisor 模式)")
    print("=" * 60)

    try:
        from langchain_core.prompts import ChatPromptTemplate  # type: ignore[import]
        from langchain_core.output_parsers import StrOutputParser  # type: ignore[import]
        from langchain.agents import create_react_agent, AgentExecutor  # type: ignore[import]
        from langchain_core.tools import tool  # type: ignore[import]
    except ImportError as e:
        print(f"  ⚠️  跳过（缺少依赖: {e}）")
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    cockpit_tool, map_tool, search_tool = _build_tools()

    # ── 子 Agent 1：座舱控制专家 ──────────────────────────────────────────
    cockpit_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是智能座舱控制专家，只处理导航、音乐、电话、车辆控制等座舱功能请求。"
                   "使用 cockpit_command 工具执行用户指令。\n\n"
                   "工具列表:\n{tools}\n工具名称: {tool_names}"),
        ("human", "{input}\n\n{agent_scratchpad}"),
    ])
    cockpit_agent = AgentExecutor(
        agent=create_react_agent(llm, [cockpit_tool], cockpit_prompt),
        tools=[cockpit_tool],
        handle_parsing_errors=True,
        max_iterations=3,
    )

    # ── 子 Agent 2：信息检索专家 ──────────────────────────────────────────
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是信息检索专家，负责搜索地图POI和互联网资讯。"
                   "根据用户需求选择 map_poi_search 或 web_search 工具。\n\n"
                   "工具列表:\n{tools}\n工具名称: {tool_names}"),
        ("human", "{input}\n\n{agent_scratchpad}"),
    ])
    research_agent = AgentExecutor(
        agent=create_react_agent(llm, [map_tool, search_tool], research_prompt),
        tools=[map_tool, search_tool],
        handle_parsing_errors=True,
        max_iterations=3,
    )

    # ── Supervisor：任务路由与结果聚合 ────────────────────────────────────
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", textwrap.dedent("""\
            你是多Agent系统的协调者。收到用户请求后，你需要：
            1. 分析请求包含哪些子任务
            2. 将座舱控制子任务（导航/音乐/电话）路由给 Cockpit Agent
            3. 将信息检索子任务（天气/地图/搜索）路由给 Research Agent
            4. 汇总两个Agent的返回结果，给出最终回答

            子任务执行结果：
            Cockpit Agent 结果: {cockpit_result}
            Research Agent 结果: {research_result}

            请用友好的语气将以上结果整合为一个完整的回答。""")),
        ("human", "用户原始请求: {user_input}"),
    ])

    def run_multi_agent(user_input: str) -> str:
        """Run both sub-agents and aggregate results via the supervisor."""
        print(f"\n  📨 用户: {user_input}")

        # 座舱子任务
        cockpit_tasks = []
        research_tasks = []

        # 简单的任务分拣（实际场景可用 LLM 做路由决策）
        cockpit_keywords = ["导航", "音乐", "播放", "电话", "空调", "开窗", "关窗"]
        research_keywords = ["天气", "搜索", "查找", "加油站", "餐厅", "新闻"]

        for kw in cockpit_keywords:
            if kw in user_input:
                cockpit_tasks.append(user_input)
                break
        for kw in research_keywords:
            if kw in user_input:
                research_tasks.append(user_input)
                break

        cockpit_result = "（未触发座舱任务）"
        research_result = "（未触发信息检索任务）"

        if cockpit_tasks:
            print("  🎛️  → Cockpit Agent 处理中...")
            try:
                res = cockpit_agent.invoke({"input": user_input})
                cockpit_result = res.get("output", "")
                print(f"  ✓  Cockpit: {cockpit_result}")
            except Exception as e:
                cockpit_result = f"执行出错: {e}"
                print(f"  ⚠️  Cockpit 错误: {e}")

        if research_tasks:
            print("  🔍  → Research Agent 处理中...")
            try:
                res = research_agent.invoke({"input": user_input})
                research_result = res.get("output", "")
                print(f"  ✓  Research: {research_result}")
            except Exception as e:
                research_result = f"执行出错: {e}"
                print(f"  ⚠️  Research 错误: {e}")

        # Supervisor 汇总
        supervisor_chain = supervisor_prompt | llm | StrOutputParser()
        final = supervisor_chain.invoke({
            "user_input": user_input,
            "cockpit_result": cockpit_result,
            "research_result": research_result,
        })
        return final

    query = "帮我导航到最近的加油站，并查一下今天北京的天气情况"
    print(f"\n用户请求: {query}")
    try:
        final_answer = run_multi_agent(query)
        print(f"\n✅ Supervisor 汇总回答:\n{final_answer}")
    except Exception as e:
        print(f"⚠️  执行出错（通常是因为未配置 OPENAI_API_KEY）: {e}")


# ---------------------------------------------------------------------------
# 方案三：LangGraph 状态图工作流
# ---------------------------------------------------------------------------

def demo_langgraph_workflow() -> None:
    """使用 ZCAgent 内置的 LangGraph 风格状态图工作流。

    不依赖 LLM API Key，展示完整的状态图执行过程。
    """
    print("\n" + "=" * 60)
    print("🗺️  方案三：LangGraph 状态图工作流（无需 LLM Key）")
    print("=" * 60)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.integrations.langgraph_adapter import create_langgraph_workflow

    workflow = create_langgraph_workflow()

    cases = [
        ("导航到天安门", "parked"),
        ("导航到天安门，顺便放首爵士乐", "parked"),
        ("看视频", "driving"),          # 行驶中被安全拦截
        ("天气怎么样", "parked"),
    ]

    for user_input, driving_state in cases:
        print(f"\n  输入: {user_input!r}  (驾驶状态: {driving_state})")
        state = workflow.invoke({"user_input": user_input, "driving_state": driving_state})
        print(f"  意图: {state.intent.get('type', 'unknown')} "
              f"(置信度 {state.intent.get('confidence', 0):.2f})")
        if state.tool_results:
            print(f"  工具调用: {list(state.tool_results.keys())}")
        print(f"  回答: {state.final_response}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    print("ZCAgent × LangChain 多 Agent 协同演示")
    print("项目: https://github.com/gqhhhhh/ZCAgent")

    # 方案三不需要 LangChain，始终演示
    demo_langgraph_workflow()

    if not _check_deps():
        print("\n💡 提示：以上 LangGraph 工作流演示已在无 LangChain 依赖下运行。")
        print("   安装 LangChain 后可体验方案一（ReAct Agent）和方案二（多 Agent 协同）。")
        return

    # 尝试初始化 LLM
    llm = None
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI  # type: ignore[import]
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            print("\n✅ 检测到 OPENAI_API_KEY，将使用真实 LLM。")
        except ImportError:
            print("\n⚠️  未安装 langchain-openai，请运行: pip install langchain-openai")
    else:
        print("\n⚠️  未设置 OPENAI_API_KEY，ReAct / 多 Agent 演示将降级为直接工具调用。")

    # 方案一：单 Agent + 多工具
    demo_single_agent(llm)

    # 方案二：多 Agent 协同（需要 LLM）
    if llm is not None:
        demo_multi_agent(llm)
    else:
        print("\n⏭️  跳过方案二（多 Agent 协同），原因：未配置 OPENAI_API_KEY。")


if __name__ == "__main__":
    main()
