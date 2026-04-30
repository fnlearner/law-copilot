"""
LawCopilot 全局配置
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """应用配置"""

    # ===== 应用基础 =====
    APP_NAME: str = "LawCopilot"
    DEBUG: bool = True

    # ===== API 服务 =====
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ===== CORS =====
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ===== LLM 配置 (DeepSeek / ds-flash) =====
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MODEL: str = "deepseek-v4-flash"
    LLM_TEMPERATURE: float = 0.1  # 法律场景低温度，确保严谨性
    LLM_MAX_TOKENS: int = 4096

    # ===== Qdrant 向量数据库 =====
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "law_copilot_laws"
    QDRANT_VECTOR_SIZE: int = 1024

    # ===== Embedding 模型 (⚠️ 不是 LLM！是专用向量模型) =====
    # Provider: local(推荐/FastEmbed) / sentence_transformer / openai_api / auto
    EMBEDDING_PROVIDER: str = "local"
    
    # 本地模型名（Provider=local 时生效，必须在 FastEmbed 白名单内）
    # ⭐ 推荐(稳定): jinaai/jina-embeddings-v2-base-zh (1024维, 8K上下文, 中英混合)
    # 轻量: BAAI/bge-small-zh-v1.5 (512维, ~90MB)
    # 效果最强(需ST): BAAI/bge-m3 (1024维, 8K上下文, 切换到 sentence_transformer)
    EMBEDDING_MODEL: str = "jinaai/jina-embeddings-v2-base-zh"
    
    # 向量维度（必须与模型匹配，创建 Collection 时用）
    EMBEDDING_DIMENSIONS: int = 1024
    
    # 云 API 配置（Provider=openai_api 时需要）
    # 推荐用智谱: https://open.bigmodel.cn/api/paas/v4  模型 embedding-3
    # 或 OpenAI: https://api.openai.com/v1  模型 text-embedding-3-small
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""

    # ===== RAG 参数 =====
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 5  # 检索返回TopK条
    SCORE_THRESHOLD: float = 0.3  # 相关度阈值

    # ===== 文档存储 =====
    DATA_DIR: str = "./data/laws"
    UPLOAD_DIR: str = "./data/uploads"

    # ===== 数据库 (SQLite) =====
    DATABASE_URL: str = "sqlite+aiosqlite:///./law_copilot.db"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
