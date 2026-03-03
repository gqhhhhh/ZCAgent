# RAG 模块 —— 检索增强生成

## 概述

`rag/` 实现了完整的 PDF 知识库构建与混合检索管线。支持从外部 PDF 文档构建知识库，通过 BM25 稀疏检索 + 向量密集检索 + ColBERT 重排序 三级混合策略进行高质量检索。

### 整体架构

```
PDF 文件
   │
   ▼
┌──────────────┐
│  PDFLoader   │──── 提取文本（按页）
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ TextChunker  │──── 递归切分（支持重叠）
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│     索引 & 存储           │
│  ┌────────┐ ┌──────────┐ │
│  │  BM25  │ │VectorStore│ │
│  │  Index │ │(NumPy)    │ │
│  └────────┘ └──────────┘ │
└──────────────────────────┘
       │
       ▼  查询时
┌──────────────────────────────────────────┐
│          混合检索管线                      │
│                                          │
│  查询 → BM25 稀疏检索 ──┐                │
│                          ├→ 加权融合       │
│  查询 → 向量密集检索 ──┘    │             │
│                              ▼            │
│                       ColBERT 重排序       │
│                              │            │
│                              ▼            │
│                         最终结果           │
└──────────────────────────────────────────┘
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `pdf_loader.py` | PDF 文档加载器：从 PDF 文件提取文本，保留页码等元数据 |
| `chunker.py` | 文本切分器：递归字符切分，支持配置 chunk_size 和 overlap |
| `vector_store.py` | 向量存储：基于 NumPy 的轻量级向量库，支持余弦相似度检索和磁盘持久化 |
| `knowledge_base.py` | 知识库管理器：一体化管线 PDF → 切分 → 索引 → 检索 → 持久化 |
| `bm25_retriever.py` | BM25 稀疏检索：基于词频-逆文档频率的经典检索算法 |
| `mmr_retriever.py` | MMR 多样性检索：在相关性和结果多样性之间取平衡 |
| `colbert_reranker.py` | ColBERT 风格重排序：基于 Token 级别 MaxSim 的细粒度相关性打分 |
| `hybrid_retriever.py` | 混合检索器：BM25 + MMR + ColBERT 三级管线，支持直接加载 PDF |

## 快速使用

### 方式一：使用 KnowledgeBase（推荐）

```python
from src.rag.knowledge_base import KnowledgeBase

# 创建知识库
kb = KnowledgeBase({
    "chunk_size": 512,       # 每个切片最大字符数
    "chunk_overlap": 64,     # 相邻切片重叠字符数
    "bm25_weight": 0.4,      # BM25 结果权重
    "vector_weight": 0.6,    # 向量检索结果权重
    "top_k": 5,              # 每个检索器返回候选数
    "rerank_top_k": 3,       # ColBERT 重排后返回数
})

# 导入 PDF
kb.add_pdf("docs/vehicle_manual.pdf")
kb.add_pdf_directory("docs/manuals/")   # 批量导入目录

# 持久化到磁盘
kb.save("data/kb_store")

# 从磁盘加载（无需重新索引）
kb2 = KnowledgeBase()
kb2.load("data/kb_store")

# 混合检索
results = kb2.search("如何设置空调温度")
for doc in results:
    print(f"[{doc.doc_id}] {doc.content[:80]}... (score={doc.score:.3f})")
```

### 方式二：使用 HybridRetriever 直接加载 PDF

```python
from src.rag.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()

# 直接从 PDF 加载（自动切分）
retriever.load_pdf("docs/vehicle_manual.pdf", chunk_size=512, chunk_overlap=64)

# 检索
results = retriever.retrieve("导航系统怎么用")
for doc in results:
    print(f"{doc.doc_id}: {doc.content} (score={doc.score:.3f})")
```

### 方式三：手动控制每一步

```python
from src.rag.pdf_loader import PDFLoader
from src.rag.chunker import TextChunker
from src.rag.vector_store import VectorStore
from src.rag.bm25_retriever import BM25Retriever

# 1. 加载 PDF
pages = PDFLoader().load("docs/vehicle_manual.pdf")

# 2. 切分
chunks = TextChunker(chunk_size=512, chunk_overlap=64).chunk_documents(pages)

# 3. 索引
bm25 = BM25Retriever()
bm25.add_documents(chunks)

vector_store = VectorStore()
vector_store.add_documents(chunks)

# 4. 检索
bm25_results = bm25.retrieve("空调温度", top_k=5)
vector_results = vector_store.search("空调温度", top_k=5)

# 5. 持久化
vector_store.save("data/vs_store")
```

## 切分策略详解

`TextChunker` 使用递归字符切分策略：

1. **优先按段落分隔** (`\n\n`)
2. **其次按换行分隔** (`\n`)
3. **再按句号等标点分隔** (`。`, `.`, `！`, `!`, `？`, `?`)
4. **最后硬切分** (按 `chunk_size` 字符数切分)

每相邻两个 chunk 之间保留 `chunk_overlap` 个字符的重叠区域，确保不丢失跨边界的上下文信息。

### 参数推荐

| 文档类型 | chunk_size | chunk_overlap |
|---------|------------|---------------|
| 技术手册 | 512 | 64 |
| 长篇文档 | 1024 | 128 |
| 短文 / FAQ | 256 | 32 |

## 配置

```yaml
knowledge_base:
  chunk_size: 512          # 每个切片最大字符数
  chunk_overlap: 64        # 相邻切片重叠字符数
  bm25_weight: 0.4         # BM25 结果权重
  vector_weight: 0.6       # 向量检索结果权重
  top_k: 5                 # 每个检索器返回候选数
  rerank_top_k: 3          # ColBERT 重排后最终返回数
  persist_directory: "data/kb_store"  # 持久化目录
```

## 技术栈

- **pypdf** — PDF 文本提取
- **Okapi BM25** — 经典信息检索算法（参数 k1=1.5, b=0.75）
- **MMR (Maximal Marginal Relevance)** — λ 参数控制相关性 vs 多样性
- **ColBERT Late Interaction** — Token 级最大相似度（MaxSim）打分
- **NumPy** — 向量运算、余弦相似度计算、持久化存储
- **jieba 分词** — 中文文本分词（可选）

> **注意**：当前 embedding 使用哈希模拟实现。生产环境请替换为真实的 embedding 模型（如 `text-embedding-ada-002` 或 `bge-large-zh`）。
