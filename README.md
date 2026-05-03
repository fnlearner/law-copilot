# LawCopilot - 法律研究助手

面向执业律师的法律研究 RAG + LLM 系统，支持法条语义检索、判例 BM25 检索、法律问答（多轮对话）。

![LawCopilot 界面](Snipaste_2026-04-30_23-24-26.png)
![法条检索](Snipaste_2026-04-30_23-24-47.png)
![法律问答](Snipaste_2026-04-30_23-28-58.png)

## 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+
- Docker（用于 Qdrant 向量数据库）
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
# 编辑 .env，填入 LLM_API_KEY
```

`.env` 关键配置说明：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | DeepSeek API 密钥 | 必填 |
| `LLM_MODEL` | LLM 模型 | `deepseek-chat` |
| `LLM_BASE_URL` | API 地址 | `https://api.deepseek.com/v1` |
| `EMBEDDING_PROVIDER` | 向量化方案（local/auto） | `local` |
| `EMBEDDING_MODEL` | 本地向量模型（FastEmbed 白名单） | `BAAI/bge-small-zh-v1.5` |
| `EMBEDDING_DIMENSIONS` | 向量维度 | `512` |
| `CAIL_DB_PATH` | 判例数据库路径（SQLite FTS5） | `/tmp/cail2018_fts.db` |

### 4. 安装后端依赖并启动

```bash
cd backend

# 创建虚拟环境（推荐 uv）
uv venv
source .venv/bin/activate

uv pip install -r requirements.txt

# 启动后端服务
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
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

### 方式一：FLK 国家法律法规数据库爬取（推荐）

```bash
cd backend
source .venv/bin/activate

# 索引模式：扫描文档列表 + 元数据
uv run python scripts/flk_scraper.py

# 全文模式：下载并提取法律条文全文
uv run python scripts/flk_scraper.py --fulltext
```

爬虫自动按 FLK 分类存储到 `data/laws_flk/`，每个分类独立目录 + `_index.json` 索引。

### 方式二：从旧数据进行向量迁移

```bash
cd backend
source .venv/bin/activate

# 从旧 Qdrant 集合迁移到新集合（含 re-embedding）
uv run python scripts/migrate_vectors.py
```

迁移使用本地 FastEmbed（bge-small-zh-v1.5, 512维）重新向量化，无需 API 调用。

### 方式三：用户文档上传

前端「文档管理」页面支持上传 PDF/Word/Markdown/TXT 文件，自动清洗、分块、向量化入库。

### 方式四：判例数据（CAIL2018）

裁判文书数据使用 SQLite FTS5 BM25 检索，入库脚本位于 `scripts/` 目录。

## 项目结构

```
law-copilot/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口
│   │   ├── config.py            # 全局配置
│   │   ├── models/
│   │   │   ├── schemas.py       # Pydantic 基础数据模型
│   │   │   └── enhanced.py      # 增强数据模型（AtomicLegalKnowledge 等）
│   │   ├── routers/
│   │   │   ├── chat.py          # 对话接口（含多轮上下文）
│   │   │   ├── search.py        # 检索接口（法条+判例）
│   │   │   └── document.py     # 文档管理接口
│   │   └── services/
│   │       ├── rag_service.py         # RAG 核心服务（三领域Prompt+双路检索+关键事实提取）
│   │       ├── embedding_service.py   # 向量化服务（FastEmbed/MPS）
│   │       ├── collection_manager.py  # Qdrant 多集合管理器
│   │       ├── query_rewriter.py      # 法律查询重写（提取法名+条文号）
│   │       ├── retriever_service.py   # 三路检索器（BM25+语义+字段）
│   │       ├── reranker_service.py    # BGE-Reranker 重排序服务
│   │       ├── judgment_service.py     # 判例检索服务（SQLite FTS5 BM25）
│   │       └── knowledge_extractor.py # 原子知识提取（LLM）
│   ├── scripts/
│   │   ├── flk_scraper.py            # FLK 国家法律法规库爬虫
│   │   ├── migrate_vectors.py         # Qdrant 存量向量迁移
│   │   ├── extract_knowledge.py       # 全量原子知识提取
│   │   ├── update_knowledge.py        # 增量知识更新
│   │   └── test_search.py             # 检索效果测试
│   ├── data/
│   │   └── laws_flk/                 # FLK 爬取数据（按分类目录）
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── pages/
│   │   │   ├── ChatPage.tsx    # 法律问答（多轮对话）
│   │   │   ├── SearchPage.tsx  # 法条检索
│   │   │   └── DocumentPage.tsx # 文档管理
│   │   └── services/api.ts
│   ├── package.json
│   └── pnpm-lock.yaml
└── README.md
```

## RAG 核心流程

```
用户提问
  │
  ▼
① 领域检测（detect_domain）
   ├─ 刑事 → CRIMINAL_PROMPT
   ├─ 经济 → ECONOMIC_PROMPT
   └─ 通用 → GENERAL_PROMPT
  │
  ▼
② 查询重写（QueryRewriter）
   提取 law_name + article_number
  │
  ▼
③ 双路检索
   ├─ 法条检索 → Qdrant 语义向量（FastEmbed COSINE）
   └─ 判例检索 → SQLite FTS5 BM25（CAIL2018 170万裁判文书）
  │
  ▼
④ 去重融合 + 重排序
  │
  ▼
⑤ LLM 生成（DeepSeek + 领域 Prompt）
  │
  ▼
⑥ 关键事实提取（每轮对话后异步提取，累积到 key_facts）
  └─ 下轮对话时 key_facts 注入 prompt「已讨论的关键法律事实」区块
```

## 多轮对话与上下文压缩

每次对话后，系统异步调用 LLM 从对话中提取关键法律事实（罪名、法条编号、刑期幅度、认定标准等），累积到 session 级别的 `key_facts` 列表。下轮对话时，已提取的关键事实会注入 prompt，使 LLM 能感知跨轮次的法律上下文，同时避免原始对话历史膨胀。

```
会话 1：故意伤害罪怎么判？
  → 提取：故意伤害罪依据刑法第234条 / 轻伤刑期三年以下 / 重伤刑期三到十年 ...
会话 2：轻伤呢？
  → prompt 注入：已讨论的关键法律事实：
    - 故意伤害罪依据刑法第234条
    - 轻伤刑期三年以下
    ...
会话 3：那重伤呢？
  → 同上上下文，LLM 可区分轻伤/重伤概念
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | React 18 + TypeScript + Ant Design 5 + Vite 6 |
| 后端 | Python 3.11 + FastAPI + Uvicorn |
| 法条检索 | Qdrant v1.12 + FastEmbed ONNX (bge-small-zh-v1.5, 512维) |
| 判例检索 | SQLite FTS5 BM25（CAIL2018 170万裁判文书） |
| LLM | DeepSeek `deepseek-chat` |
| 重排序 | Jina Reranker API |
| 文档处理 | LangChain + Unstructured |

## 环境变量参考

```bash
# DeepSeek LLM
LLM_API_KEY=***
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_TEMPERATURE=0.1

# Embedding（本地模型，无需 API Key）
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DIMENSIONS=512

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=laws

# 判例数据库（SQLite FTS5）
CAIL_DB_PATH=/tmp/cail2018_fts.db

# RAG 参数
TOP_K=5
SCORE_THRESHOLD=0.3
```

## License

MIT
