# LawCopilot 系统设计

## 1. 项目概述

**目标**：面向执业律师的经济类法律研究助手，通过 RAG + LLM 帮助律师快速检索法条、分析案例、生成法律分析。

**核心能力**：
- 语义法条检索（无需精确关键词匹配）
- 基于法律法规的智能法律问答
- 自动引用来源并标注相关度
- 支持多种任务类型：法条检索 / 案例分析 / 文书生成 / 深度研究

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                        用户层                                  │
│              Web 浏览器 (http://localhost:3000)               │
└─────────────────────────┬────────────────────────────────────┘
                          │ HTTP / JSON
┌─────────────────────────▼────────────────────────────────────┐
│                      后端层 (FastAPI)                         │
│                                                              │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
│   │   Chat API   │   │  Search API  │   │ Document API │  │
│   │  /chat/ask   │   │ /search/query│  │ /document/*  │  │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘  │
│          └───────────────────┼───────────────────┘          │
│                              │                                │
│                    ┌─────────▼─────────┐                     │
│                    │   RAG Service    │                     │
│                    │  (检索 + 生成)    │                     │
│                    └─────────┬─────────┘                     │
│                              │                                │
│          ┌───────────────────┼───────────────────┐          │
│          │                   │                   │          │
│  ┌───────▼───────┐  ┌───────▼───────┐  ┌──────▼──────┐   │
│  │ Embedding      │  │ Qdrant        │  │ DeepSeek    │   │
│  │ Service        │  │ Vector Store  │  │ LLM API     │   │
│  │ (Jina AI)      │  │ (:6333)       │  │             │   │
│  └───────────────┘  └───────────────┘  └──────────────┘   │
└──────────────────────────────────────────────────────────────┘

数据来源层：
  LawRefBook/Laws (GitHub) → 解析入库 → Qdrant 向量库
```

## 3. 核心数据流

### 3.1 文档入库流程

```
LawRefBook/Laws (.md)
        │
        ▼
 chunk_file() — 按「第X条」正则切分
        │
        ▼
 metadata() — 提取 doc_type, law_name, article_number
        │
        ▼
 Jina AI API (jina-embeddings-v3)
   → 1024 维归一化向量
        │
        ▼
 Qdrant upsert (batch=32)
        │
        ▼
 Collection: law_copilot_laws
   - vector: float[1024]
   - payload: { doc_type, law_name, article_number,
                 source_file, char_length, ... }
```

### 3.2 问答流程（Chat / RAG）

```
用户提问
   │
   ▼
 Jina AI API — 将问题向量化 (query embedding)
   │
   ▼
 Qdrant search — 余弦相似度检索 top_k=5
   │
   ▼
 拼装 Prompt — 问题 + 检索结果(法条内容) + System Prompt
   │
   ▼
 DeepSeek LLM (deepseek-chat)
   │
   ▼
 返回：AI 分析文本 + 引用来源列表
```

## 4. 技术选型与理由

| 组件 | 选型 | 原因 |
|------|------|------|
| **Embedding** | Jina AI `jina-embeddings-v3` | 1024 维，支持中文，API 调用无需本地 GPU |
| **Vector DB** | Qdrant v1.12 | 原生支持 cosine 距离、payload 过滤、性能优秀 |
| **LLM** | DeepSeek `deepseek-chat` | 性价比高，中文法律文本理解能力强 |
| **后端框架** | FastAPI | 异步、高性能、自动生成 OpenAPI 文档 |
| **前端框架** | React 18 + Ant Design 5 | 组件丰富，适合中台类 Admin 系统 |
| **构建工具** | Vite 6 | HMR 快，开发体验好 |
| **前端包管理** | pnpm | 比 npm 更快、更省空间 |
| **向量入库** | Python 虚拟环境 (uv) | 轻量、快速、可重现 |

### 为什么不用 LangChain 做向量存储？

最初尝试用 `langchain-qdrant` + `LangChainEmbeddingsAdapter`，但适配层在运行时验证失败。最终改用 **Qdrant 原生 Python 客户端**直接做 upsert / search，跳过 LangChain 向量存储层，代码更简洁、稳定性更高。

### 为什么用 Jina AI 而不是本地模型？

- 本地 BGE-M3 (MPS) 速度约 9 texts/s，预计 4.6 小时入库 148K 条
- Jina AI API 约 15-25 texts/s，API 稳定，无需管理模型文件
- SiliconFlow 等第三方 API 存在维度不兼容问题（4096 vs 1024）

## 5. 模块说明

### 5.1 后端模块

```
backend/app/
├── main.py                 # FastAPI 入口，路由注册， lifespan 管理
├── config.py               # 环境变量解析，Pydantic Settings
├── models/schemas.py       # 请求/响应 Pydantic 模型
├── routers/
│   ├── chat.py             # /chat/ask, /chat/stream
│   ├── search.py           # /search/query, /search/suggestions, /search/stats
│   └── document.py          # /document/upload, /document/import-laws, /document/seed-demo
└── services/
    ├── rag_service.py      # 核心 RAG：search() + generate()
    ├── embedding_service.py # Embedding 服务（Jina AI API 调用）
    └── document_pipeline.py # 文档处理 pipeline（上传/清洗/分块/入库）
```

### 5.2 前端模块

```
frontend/src/
├── App.jsx                 # 主布局（Header + Content）
├── main.jsx                # React 入口
├── pages/
│   ├── ChatPage.jsx        # 法律问答页面（核心）
│   ├── SearchPage.jsx      # 法条检索页面
│   └── DocumentPage.jsx    # 文档管理页面
└── services/api.js         # axios 封装，统一 HTTP 调用
```

## 6. API 设计

### 6.1 对话接口

**POST /api/v1/chat/ask**

Request:
```json
{
  "message": "劳动合同解除的赔偿标准是什么？",
  "task_type": "legal_research",
  "scope": "economic",
  "top_k": 5
}
```

Response:
```json
{
  "reply": "根据《劳动合同法》第四十六条...",
  "references": [
    {
      "doc_type": "labor_law",
      "title": "中华人民共和国劳动合同法",
      "content_snippet": "第四十六条 有下列情形之一的...",
      "relevance_score": 0.87,
      "article_number": "第四十六条"
    }
  ],
  "session_id": "uuid",
  "task_type": "legal_research",
  "model_used": "deepseek-chat",
  "latency_ms": 2340
}
```

### 6.2 检索接口

**POST /api/v1/search/query**

Request:
```json
{
  "query": "公司股东股权转让程序",
  "scope": "all",
  "top_k": 5
}
```

### 6.3 文档管理接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/document/upload` | 上传文档（PDF/Word/MD/TXT） |
| POST | `/api/v1/document/import-laws` | 从 LawRefBook 目录批量导入 |
| POST | `/api/v1/document/seed-demo` | 写入示范法条（演示用） |
| GET | `/api/v1/document/list` | 列出已入库文档 |

## 7. 数据模型

### Qdrant Collection: `law_copilot_laws`

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | integer | 自增主键 |
| `vector` | float[1024] | jina-embeddings-v3 归一化向量 |
| `payload.doc_type` | string | 法规类型：civil/criminal/administrative/economic/... |
| `payload.law_name` | string | 法规名称，如《中华人民共和国公司法》 |
| `payload.article_number` | string | 条文编号，如"第一百三十二条" |
| `payload.source_file` | string | 源文件路径 |
| `payload.char_length` | integer | 条文字符数 |
| `payload.file_name` | string | 文件名 |
| `payload.chapter` | string | 章（暂未使用） |

**索引配置**：
- `vectors.size`: 1024
- `vectors.distance`: Cosine
- `hnsw_config.m`: 16
- `hnsw_config.ef_construct`: 100

## 8. RAG 参数

| 参数 | 值 | 说明 |
|------|------|------|
| `TOP_K` | 5 | 检索返回条数 |
| `SCORE_THRESHOLD` | 0.3 | 最低相关度阈值 |
| `CHUNK_SIZE` | 512 | 文本分块大小（字符数） |

## 9. 部署架构

### 开发模式

```
终端 1 (后端):
  cd backend
  uv venv && uv pip install -r requirements.txt
  uvicorn app.main:app --host 0.0.0.0 --port 8000

终端 2 (前端):
  cd frontend
  pnpm install && pnpm run dev

终端 3 (Qdrant):
  docker run -d --name law-qdrant \
    -p 6333:6333 -p 6334:6334 \
    -v ~/docker/qdrant:/qdrant/storage \
    qdrant/qdrant:v1.12.0
```

### 生产模式（Docker）

```bash
# 后端
docker build -t law-copilot-backend ./backend
docker run -d -p 8000:8000 --name law-backend law-copilot-backend

# 前端
docker build -t law-copilot-frontend ./frontend
docker run -d -p 3000:80 law-copilot-frontend

# Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.12.0
```

## 10. 已实现功能 vs 未完成

### ✅ 已完成

- [x] Qdrant 法律条文向量库（46K+ 条）
- [x] Jina AI API 向量化（1024 维）
- [x] RAG 检索 + DeepSeek LLM 生成
- [x] 前端法律问答界面（ChatPage）
- [x] 前端法条检索界面（SearchPage）
- [x] 文档管理界面（DocumentPage）
- [x] 批量入库脚本（ingest_laws_jina.py）
- [x] 增量入库脚本（ingest_laws_jina_incremental.py）
- [x] Swagger API 文档（/docs）

### 🔲 待完成

- [ ] Phase 1 Agents：schemas.py、registry.py、base.py、4 个领域专家 Agent、coordinator
- [ ] 流式对话（/chat/stream 接口已写好，前端未对接）
- [ ] 案例分析任务类型（analyze_case）
- [ ] 文书生成任务类型（generate_doc）
- [ ] 向量库完整入库（当前 46K/148K）
- [ ] 搜索结果高亮和展开
- [ ] 用户认证与会话管理
- [ ] 性能优化：Qdrant HNSW 参数调优
