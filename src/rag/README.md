# RAG 模块 —— 检索增强生成

## 概述

`rag/` 实现了三级混合检索管线，为 Agent 提供外部知识增强能力：

```
查询 → BM25 稀疏检索 ──┐
                         ├──→ 加权融合 → ColBERT 重排序 → 最终结果
查询 → MMR 多样性检索 ──┘
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `bm25_retriever.py` | BM25 稀疏检索：基于词频-逆文档频率的经典检索算法，擅长精确关键词匹配 |
| `mmr_retriever.py` | MMR 多样性检索：在相关性和结果多样性之间取平衡，避免重复信息 |
| `colbert_reranker.py` | ColBERT 风格重排序：基于 Token 级别 MaxSim 的细粒度相关性打分 |
| `hybrid_retriever.py` | 混合检索器：组合 BM25 + MMR，经 ColBERT 重排后返回最终结果 |

## 技术栈

- **Okapi BM25** — 经典信息检索算法（参数 k1=1.5, b=0.75）
- **MMR (Maximal Marginal Relevance)** — λ 参数控制相关性 vs 多样性的权衡
- **ColBERT Late Interaction** — Token 级最大相似度（MaxSim）打分
- **jieba 分词** — 中文文本分词（可选，不安装时退化为字符级分词）
- **NumPy** — 向量运算和余弦相似度计算

> **注意**：当前 embedding 使用哈希模拟实现。生产环境请替换为真实的 embedding 模型（如 `text-embedding-ada-002` 或 `bge-large-zh`）。

## 快速使用

```python
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.bm25_retriever import Document

retriever = HybridRetriever()

# 添加文档
docs = [
    Document(doc_id="1", content="天安门广场位于北京市中心"),
    Document(doc_id="2", content="故宫博物院在天安门北侧"),
    Document(doc_id="3", content="上海外滩是著名景点"),
]
retriever.add_documents(docs)

# 检索
results = retriever.retrieve("北京天安门附近景点")
for doc in results:
    print(f"{doc.doc_id}: {doc.content} (score={doc.score:.3f})")
```

## 配置

```yaml
rag:
  bm25_weight: 0.4        # BM25 结果权重
  mmr_weight: 0.6          # MMR 结果权重
  mmr_lambda: 0.7          # MMR 相关性 vs 多样性权衡（0=纯多样性, 1=纯相关性）
  top_k: 5                 # 每个检索器返回候选数
  rerank_top_k: 3          # ColBERT 重排后最终返回数
```
