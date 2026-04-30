# LawCopilot - 法律研究助手

面向执业律师的经济类法律研究 RAG + LLM 系统，支持法条检索、案例分析、法律文书生成。

## 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+
- Docker（用于 Qdrant 向量数据库）
- Jina AI API Key（[申请地址](https://jina.ai/embeddings/)）
- DeepSeek API Key（[申请地址](https://platform.deepseek.com/)）

### 1. 克隆项目

```bash
git clone <your-repo> law-copilot
cd law-copilot
```

### 2. 启动 Qdrant 向量数据库

```bash
docker run -d \
  --name law-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v ~/docker/qdrant:/qdrant/storage \
  qdrant/qdrant:v1.12.0
```

### 3. 配置后端

```bash
cd backend
cp .env.example .env
# 编辑 .env，填入以下必填项：
# LLM_API_KEY=your-deepseek-api-key
# EMBEDDING_API_KEY=your-jina-ai-api-key
```

`.env` 关键配置说明：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | DeepSeek API 密钥 | 必填 |
| `LLM_MODEL` | LLM 模型 | `deepseek-chat` |
| `LLM_BASE_URL` | API 地址 | `https://api.deepseek.com/v1` |
| `EMBEDDING_API_KEY` | Jina AI API 密钥 | 必填 |
| `EMBEDDING_MODEL` | Embedding 模型 | `jina-embeddings-v3` |
| `EMBEDDING_DIMENSIONS` | 向量维度 | `1024` |

### 4. 安装后端依赖并启动

```bash
cd backend

# 创建虚拟环境（推荐 uv）
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

uv pip install -r requirements.txt

# 启动后端服务
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. 安装前端依赖并启动

```bash
cd frontend

pnpm install
pnpm run dev
```

### 6. 访问

- 前端：**http://localhost:3000**
- 后端 API：**http://localhost:8000**
- Qdrant Dashboard：**http://localhost:6333/dashboard**
- API 文档：**http://localhost:8000/docs**（Swagger UI）

## 法律数据入库

### 方式一：Jina AI API 批量入库（推荐）

```bash
cd backend
source .venv/bin/activate

# 克隆法律数据
git clone https://github.com/LawRefBook/Laws.git /tmp/Laws

# 批量入库（需配置好 .env 中的 Jina API Key）
uv run python scripts/ingest_laws_jina.py
```

入库脚本会：
1. 扫描 `/tmp/Laws` 目录下的所有 .md 法律文件
2. 按"第X条"正则切分法律条文
3. 调用 Jina AI `jina-embeddings-v3` API 生成 1024 维向量
4. 写入本地 Qdrant 数据库

预计耗时：15-30 分钟（取决于 API 速率）

### 方式二：写入示范数据（快速体验）

在前端「文档管理」页面点击「写入示范法条」，或调用接口：

```bash
curl -X POST http://localhost:8000/api/v1/document/seed-demo
```

### 方式三：用户文档上传

前端「文档管理」页面支持上传 PDF/Word/Markdown/TXT 文件，自动清洗、分块、向量化入库。

## 项目结构

```
law-copilot/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口
│   │   ├── config.py            # 全局配置
│   │   ├── models/schemas.py    # Pydantic 数据模型
│   │   ├── routers/            # API 路由
│   │   │   ├── chat.py         # 对话接口
│   │   │   ├── search.py       # 检索接口
│   │   │   └── document.py     # 文档管理接口
│   │   └── services/
│   │       ├── rag_service.py   # RAG 核心服务
│   │       ├── embedding_service.py
│   │       └── document_pipeline.py
│   ├── scripts/
│   │   ├── ingest_laws_jina.py       # 法律条文批量入库
│   │   └── test_rag.py                # RAG 效果测试
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── pages/
│   │   │   ├── ChatPage.jsx    # 法律问答
│   │   │   ├── SearchPage.jsx  # 法条检索
│   │   │   └── DocumentPage.jsx # 文档管理
│   │   └── services/api.js
│   ├── package.json
│   └── pnpm-lock.yaml
└── README.md
```

## 技术架构

```
┌─────────────────────────────────────┐
│    前端 (React + Ant Design)        │
│    http://localhost:3000            │
└──────────────┬──────────────────────┘
               │ HTTP
┌──────────────▼──────────────────────┐
│    后端 (FastAPI)                   │
│    http://localhost:8000            │
│                                      │
│  ┌────────────┐  ┌──────────────┐  │
│  │ 对话路由    │  │  检索路由      │  │
│  └──────┬─────┘  └──────┬───────┘  │
│         └───────────────┼───────────┘
│                     RAG 服务
│              检索 + LLM 生成
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    │                      │
┌───▼────────┐    ┌───────▼──────┐
│  Qdrant     │    │  DeepSeek    │
│  向量数据库   │    │  LLM API     │
│  :6333      │    │  (对话生成)   │
└─────────────┘    └──────────────┘

Embedding 流程（独立）：
Jina AI API (jina-embeddings-v3)
→ Qdrant 向量库
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | React 18 + Ant Design 5 + Vite 6 |
| 后端 | Python 3.11 + FastAPI |
| 向量数据库 | Qdrant v1.12 |
| Embedding | Jina AI `jina-embeddings-v3`（1024 维） |
| LLM | DeepSeek `deepseek-chat` |
| 文档处理 | LangChain + Unstructured |

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat/ask` | 法律问答（核心） |
| POST | `/api/v1/chat/stream` | 流式对话 |
| POST | `/api/v1/search/query` | 法条语义检索 |
| GET | `/api/v1/search/suggestions?q=` | 搜索建议 |
| POST | `/api/v1/document/upload` | 上传文档 |
| POST | `/api/v1/document/import-laws` | 批量导入法律库 |
| POST | `/api/v1/document/seed-demo` | 写入示范法条 |
| GET | `/api/v1/health` | 健康检查 |

## 环境变量参考

完整配置见 `backend/.env.example`，关键变量：

```bash
# DeepSeek LLM
LLM_API_KEY=sk-xxxx
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_TEMPERATURE=0.1

# Jina AI Embedding
EMBEDDING_API_KEY=jina_xxxx
EMBEDDING_MODEL=jina-embeddings-v3
EMBEDDING_DIMENSIONS=1024
EMBEDDING_BASE_URL=https://api.jina.ai/v1

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=law_copilot_laws

# RAG 参数
CHUNK_SIZE=512
TOP_K=5
SCORE_THRESHOLD=0.3
```

## License

MIT
