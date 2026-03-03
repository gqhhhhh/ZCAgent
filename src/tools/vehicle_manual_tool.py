"""Vehicle Manual RAG tool for active retrieval-based Q&A.

车辆使用手册 RAG 工具：基于混合检索（BM25 + MMR + ColBERT）的主动检索工具。
内置车辆使用手册知识库，Agent 可通过此 Tool 主动查询车辆功能、操作说明、
安全须知等信息，并返回最相关的手册段落作为回答依据。
"""

import logging

from src.rag.bm25_retriever import Document
from src.rag.hybrid_retriever import HybridRetriever
from src.tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# 内置车辆使用手册文档（涵盖驾驶安全、空调、导航、音乐、电话、车窗、灯光、
# 座椅、轮胎、电池等主要功能领域）
_DEFAULT_MANUAL_DOCUMENTS = [
    Document("vm001", "车辆启动前请确保所有车门已关闭并系好安全带。点火启动时，踩住刹车踏板并按下一键启动按钮。",
             {"category": "驾驶安全"}),
    Document("vm002", "当车速超过120km/h时，禁止打开车窗以确保行车安全。高速行驶时建议使用空调系统代替开窗通风。",
             {"category": "驾驶安全"}),
    Document("vm003", "紧急制动辅助系统（EBA）在检测到紧急制动时会自动增加制动力，帮助缩短制动距离。该系统在车速高于10km/h时自动激活。",
             {"category": "驾驶安全"}),
    Document("vm004", "车道偏离预警系统在车速超过60km/h时自动启用。如果车辆偏离车道且未打转向灯，系统会通过方向盘振动和声音提醒驾驶员。",
             {"category": "驾驶安全"}),
    Document("vm005", "空调系统可以设置温度范围16-32度，支持AUTO自动模式。自动模式下系统根据车内外温差自动调节风量和出风方向。",
             {"category": "空调系统"}),
    Document("vm006", "座椅加热分为三档：低档、中档和高档。建议在寒冷天气下先使用高档预热，待座椅温暖后切换至中档或低档。",
             {"category": "座椅系统"}),
    Document("vm007", "导航系统支持语音输入目的地，也可以手动输入地址或从收藏夹中选择。导航可以设置途经点，最多支持3个途经点。",
             {"category": "导航系统"}),
    Document("vm008", "导航系统提供三种路径偏好：最短时间、最短距离和避免高速。实时路况功能需要连接网络才能使用。",
             {"category": "导航系统"}),
    Document("vm009", "音乐播放器支持蓝牙音频、USB音源和在线音乐流媒体三种播放方式。音量可通过方向盘按键或中控屏幕调节。",
             {"category": "娱乐系统"}),
    Document("vm010", "蓝牙电话功能支持自动连接已配对手机。拨打电话时可通过语音说出'打电话给XXX'进行免提拨号。通话中可切换为车载扬声器或手机听筒。",
             {"category": "电话系统"}),
    Document("vm011", "天窗在车速超过100km/h时自动限制开启角度以降低风噪。天窗带有防夹功能，在关闭时如检测到障碍物会自动回弹。",
             {"category": "车窗天窗"}),
    Document("vm012", "自动大灯功能可根据环境光线自动切换近光灯和示宽灯。进入隧道时自动开启大灯，驶出后自动关闭。",
             {"category": "灯光系统"}),
    Document("vm013", "远光灯辅助系统在夜间行驶时自动检测对向来车和前方车辆，智能切换远近光灯，避免造成眩光。",
             {"category": "灯光系统"}),
    Document("vm014", "胎压监测系统实时显示四个轮胎的气压和温度。正常胎压范围为2.3-2.5bar。胎压低于2.0bar时系统会发出警报。",
             {"category": "轮胎系统"}),
    Document("vm015", "电动车辆电池在温度低于-10°C时充电速度会降低。建议在行驶结束后立即充电以利用电池余温提高充电效率。",
             {"category": "电池充电"}),
    Document("vm016", "快充模式下电池可在30分钟内从20%充至80%。为延长电池寿命，建议日常使用慢充，并将电量维持在20%-80%之间。",
             {"category": "电池充电"}),
    Document("vm017", "驻车辅助系统可自动识别平行车位和垂直车位。激活方式：在低于30km/h时按下泊车辅助按钮，系统将自动搜索可用车位。",
             {"category": "泊车辅助"}),
    Document("vm018", "360度全景影像在挂入倒挡时自动激活，也可通过中控屏幕上的摄像头按钮手动开启。可在俯视图、前视图和侧视图间切换。",
             {"category": "泊车辅助"}),
    Document("vm019", "自适应巡航控制（ACC）可在30-150km/h范围内使用。通过方向盘上的+/-按钮调节设定速度，SET按钮激活巡航。",
             {"category": "巡航控制"}),
    Document("vm020", "紧急呼叫功能在任何驾驶状态下都可以使用。按住SOS按钮3秒即可接通紧急救援中心。碰撞发生后系统可自动拨打紧急电话。",
             {"category": "紧急救援"}),
]


class VehicleManualTool(BaseTool):
    """Vehicle Manual RAG tool for answering vehicle-related questions.

    基于混合检索的车辆使用手册问答工具。内置车辆手册知识库，Agent 可主动
    调用此 Tool 查询车辆功能说明、操作方法和安全须知，返回最相关的手册内容。
    """

    name = "vehicle_manual"
    description = "车辆使用手册问答工具，可查询车辆功能说明、操作方法和安全须知"

    def __init__(self, rag_config: dict | None = None,
                 documents: list[Document] | None = None):
        """Initialize the Vehicle Manual Tool.

        Args:
            rag_config: Configuration dict for the HybridRetriever.
            documents: Optional custom documents. Uses built-in manual if None.
        """
        self._retriever = HybridRetriever(rag_config)
        docs = documents if documents is not None else _DEFAULT_MANUAL_DOCUMENTS
        if docs:
            self._retriever.add_documents(docs)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, query: str = "", top_k: int = 3, **_kwargs) -> ToolResult:
        """Retrieve relevant vehicle manual passages for the given query.

        Args:
            query: The user's question about the vehicle.
            top_k: Number of passages to return.

        Returns:
            ToolResult containing the most relevant manual passages.
        """
        if not query:
            return ToolResult(success=False, error="缺少查询问题")

        try:
            results = self._retriever.retrieve(query, rerank_top_k=top_k)
            if not results:
                return ToolResult(success=True, data={
                    "answer": "未在手册中找到相关信息。",
                    "passages": [],
                    "count": 0,
                })

            passages = []
            for doc in results:
                passages.append({
                    "doc_id": doc.doc_id,
                    "content": doc.content,
                    "category": doc.metadata.get("category", ""),
                    "score": round(doc.score, 4),
                })

            answer = passages[0]["content"]
            return ToolResult(success=True, data={
                "answer": answer,
                "passages": passages,
                "count": len(passages),
            })

        except Exception as exc:
            logger.error("Vehicle manual retrieval error: %s", exc)
            return ToolResult(success=False, error=str(exc))

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "关于车辆的查询问题",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的手册段落数量",
                    "default": 3,
                },
            },
        }
