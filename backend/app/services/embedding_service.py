"""
LawCopilot Embedding 服务 — 多后端 Embedding 模型管理器

支持的后端（按优先级）：
  1. FastEmbed (本地) — 推荐，基于 Rust ONNX Runtime，极速推理
     - jinaai/jina-embeddings-v2-base-zh → 1024维，8K上下文，中英混合，⭐推荐
     - BAAI/bge-small-zh-v1.5       → 512维，纯中文，轻量(~90MB)
     - intfloat/multilingual-e5-large  → 1024维，多语言
  2. SentenceTransformers (本地) — 备选，支持更多模型(含BGE-M3/BGE-large)
     - 可用任意 HuggingFace 模型
  3. OpenAI-Compatible API (云端) — 备用
     - 智谱 AI embedding-3 / OpenAI text-embedding-3-* 等

⚠️ 重要：FastEmbed 原生支持列表有限（截至 v0.8.0），
    中文可用：jina-embeddings-v2-base-zh / bge-small-zh-v1.5
    如需 BGE-M3 或 bge-large-zh-v1.5，请切换到 sentence_transformer 后端

设计原则：
  - 统一接口：embed_texts() / embed_query() / get_dimensions()
  - 自动降级：配置了本地模型优先用本地，失败可切云 API
  - 批量优化：自动分批（batch_size），避免 OOM
"""

import os
import logging
import hashlib
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import asyncio
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# Embedding 后端抽象接口
# ============================================================

class EmbeddingBackend(ABC):
    """Embedding 后端抽象基类"""

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化（用于文档入库）"""
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """单条查询向量化（用于检索）"""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """向量维度"""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """后端名称标识"""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        ...


# ============================================================
# 后端 1: FastEmbed (本地，Rust 加速，推荐)
# ============================================================

class FastEmbedBackend(EmbeddingBackend):
    """
    基于 FastEmbed 的本地 Embedding 推理
    
    优势：
      - 纯 Python 调用，底层 Rust ONNX Runtime，速度快
      - 无需 GPU，CPU 推理即可满足中小规模需求
      - 自动下载并缓存模型文件到 ~/.cache/fastembed/
      - 内存占用低，适合 Docker 容器
    
    ⚠️ 注意：FastEmbed 有原生模型白名单（TextEmbedding.list_supported_models()），
       不在白名单的模型会抛 ValueError。
       中文场景推荐模型：
    
    模型名                                          | 维度 | 最大长度 | 大小(约)  | 说明
    -----------------------------------------------|------|---------|----------|------
    jinaai/jina-embeddings-v2-base-zh              | 1024 | 8192    | ~400MB   | ⭐ 推荐！中英混合+长上下文
    BAAI/bge-small-zh-v1.5                         | 512  | 512     | ~90MB    | 最轻量纯中文
    intfloat/multilingual-e5-large                  | 1024 | 512     | ~1.2GB   | 多语言
    """

    # 模型名 → 维度 映射表（仅限 FastEmbed 原生支持的模型）
    MODEL_DIMENSIONS = {
        # ⭐ 中文推荐（FastEmbed 原生支持）
        "jinaai/jina-embeddings-v2-base-zh": 1024,   # ★ 默认推荐：中英混合, 8K上下文
        "BAAI/bge-small-zh-v1.5": 512,               # 最轻量纯中文
        
        # 英文/多语言（FastEmbed 原生）
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
        
        # Jina 系列
        "jinaai/jina-embeddings-v3": 1024,            # 多任务多语言, 8K
        "jinaai/jina-embeddings-v2-small-en": 512,
        "jinaai/jina-embeddings-v2-base-en": 768,
        "jinaai/jina-embeddings-v2-base-de": 768,
        "jinaai/jina-embeddings-v2-base-code": 768,
        "jinaai/jina-embeddings-v2-base-es": 768,
        
        # 其他
        "thenlper/gte-base": 768,
        "thenlper/gte-large": 1024,
        "intfloat/multilingual-e5-large": 1024,
        "nomic-ai/nomic-embed-text-v1.5": 768,
        "mixedbread-ai/mxbai-embed-large-v1": 1024,
        
        # ❌ 以下模型不在 FastEmbed 白名单中，需要用 sentence_transformer 后端：
        # "BAAI/bge-large-zh-v1.5": 1024,      ← 用 ST 后端
        # "BAAI/bge-m3": 1024,                 ← 用 ST 后端
    }

    # 中文法律场景默认模型 — 必须在 FastEmbed 原生白名单内
    DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-zh"

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
    ):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._batch_size = batch_size
        self._max_length = max_length
        self._cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "fastembed")
        self._model = None
        self._dimensions = self.MODEL_DIMENSIONS.get(self._model_name, 1024)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def backend_name(self) -> str:
        return f"fastembed({self._model_name})"

    async def _ensure_model(self):
        """懒加载模型（首次调用时才下载和加载）"""
        if self._model is not None:
            return

        try:
            import fastembed
        except ImportError:
            raise ImportError(
                "需要安装 fastembed: pip install fastembed\n"
                "推荐安装 GPU 版本以获得更好性能: pip install fastembed[gpu]"
            )

        logger.info(f"🔄 正在加载 FastEmbed 模型: {self._model_name} ...")
        
        # 在线程池中执行同步的模型加载（避免阻塞事件循环）
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: fastembed.TextEmbedding(
                model_name=self._model_name,
                max_length=self._max_length,
                cache_dir=self._cache_dir,
            ),
        )
        logger.info(f"✅ FastEmbed 模型加载完成: {self._model_name} (dim={self._dimensions})")

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量向量化"""
        await self._ensure_model()

        if not texts:
            return []

        all_embeddings = []

        # 分批处理，避免一次性送入过多文本导致内存溢出
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            
            # FastEmbed 是同步的，放线程池跑
            loop = asyncio.get_event_loop()
            batch_result = await loop.run_in_executor(
                None, list, self._model.embed(batch)
            )
            
            # 将 numpy 数组转为普通 list
            batch_embeddings = [vec.tolist() for vec in batch_result]
            all_embeddings.extend(batch_embeddings)

        logger.debug(f"FastEmbed 批量编码: {len(texts)} 条文本 → {len(all_embeddings)} 个向量")
        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        """单条查询向量化"""
        results = await self.embed_texts([query])
        return results[0] if results else []

    async def health_check(self) -> bool:
        try:
            await self._ensure_model()
            test_vec = await self.embed_query("测试")
            return len(test_vec) == self._dimensions
        except Exception as e:
            logger.warning(f"FastEmbed 健康检查失败: {e}")
            return False


# ============================================================
# 后端 2: SentenceTransformers (本地，Python 原生)
# ============================================================

class SentenceTransformerBackend(EmbeddingBackend):
    """
    基于 sentence-transformers 的本地 Embedding
    
    比 FastEmbed 支持更多模型，但速度较慢、内存占用更高（需要 PyTorch）。
    适合需要高级特性或使用不在 FastEmbed 白名单内的模型。
    
    推荐场景：
      - 使用 BGE-M3 / bge-large-zh-v1.5 等非 FastEmbed 白名单的中文模型
      - 需要 prompt 前缀指令微调
    """

    MODEL_DIMENSIONS = {
        # ⭐ 中文法律推荐（ST 专用，FastEmbed 不支持这些）
        "BAAI/bge-m3": 1024,                          # 效果最强，8K上下文
        "BAAI/bge-large-zh-v1.5": 1024,               # 经典纯中文
        "BAAI/bge-large-zh-nolist-v1.5": 1024,
        "shibing624/text2vec-large-chinese": 768,
        "shibing624/text2vec-base-chinese": 768,
    }

    DEFAULT_MODEL = "BAAI/bge-large-zh-v1.5"

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._device = device
        self._model = None
        self._dimensions = self.MODEL_DIMENSIONS.get(self._model_name, 768)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def backend_name(self) -> str:
        return f"sentence-transformer({self._model_name})"

    async def _ensure_model(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "需要安装 sentence-transformers: "
                "pip install sentence-transformers"
            )

        logger.info(f"🔄 正在加载 SentenceTransformer 模型: {self._model_name} ...")
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer(self._model_name, device=self._device),
        )
        logger.info(f"✅ SentenceTransformer 模型加载完成: {self._model_name}")

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        await self._ensure_model()

        if not texts:
            return []

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, normalize_embeddings=True).tolist(),
        )
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        results = await self.embed_texts([query])
        return results[0] if results else []

    async def health_check(self) -> bool:
        try:
            await self._ensure_model()
            test_vec = await self.embed_query("测试")
            return len(test_vec) == self._dimensions
        except Exception as e:
            logger.warning(f"SentenceTransformer 健康检查失败: {e}")
            return False


# ============================================================
# 后端 3: OpenAI-Compatible API (云端)
# ============================================================

class OpenAICompatibleBackend(EmbeddingBackend):
    """
    基于 OpenAI 兼容 API 的云端 Embedding
    
    支持的服务商：
      - OpenAI (text-embedding-3-small, text-embedding-3-large)
      - 智谱 AI (embedding-3) ← 推荐，与 LawCopilot LLM 同生态
      - 通义千问 (text-embedding-v3)
      - Azure OpenAI
      - 其他兼容 OpenAI 格式的 API
    
    优点：无需本地 GPU/内存，即开即用
    缺点：有网络延迟、有 API 费用、数据出域
    """

    # 已知模型 → 维度 映射
    KNOWN_MODELS = {
        # OpenAI
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        # 智谱 AI
        "embedding-3": 1024,
        "embedding-2": 1024,
        # 通义千问
        "text-embedding-v3": 1024,
    }

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        model: str = "embedding-3",
        dimensions: int = 1024,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions = self.KNOWN_MODELS.get(model, dimensions)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def backend_name(self) -> str:
        return f"openai-api({self._model}@{self._base_url.split('//')[-1].split('/')[0]})"

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        import httpx

        url = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        all_embeddings = []
        # OpenAI API 单次最多支持 2048 条，这里保守用 512
        batch_size = 512
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {
                "model": self._model,
                "input": batch,
                "encoding_format": "float",
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            # 按 input 顺序排列结果
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            all_embeddings.extend([item["embedding"] for item in sorted_data])

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        results = await self.embed_texts([query])
        return results[0] if results else []

    async def health_check(self) -> bool:
        try:
            test_vec = await self.embed_query("hello")
            return len(test_vec) > 0
        except Exception as e:
            logger.warning(f"OpenAI API 健康检查失败: {e}")
            return False


# ============================================================
# 主服务：EmbeddingService (统一入口 + 自动选择)
# ============================================================

class EmbeddingService:
    """
    Embedding 服务统一管理器
    
    选择逻辑（按优先级）：
    
    1. 如果 EMBEDDING_PROVIDER=local 且 EMBEDDING_MODEL 指定了本地模型名
       → 使用 FastEmbed 本地推理（推荐）
       
    2. 如果 EMBEDDING_PROVIDER=sentence_transformer
       → 使用 SentenceTransformers 本地推理
       
    3. 如果 EMBEDDING_PROVIDER=openai_api 或以上都不行
       → 使用 OpenAI 兼容 API 云端推理
    
    配置方式（通过 .env 或 Settings）：
      EMBEDDING_PROVIDER=local          # local / sentence_transformer / openai_api / auto
      EMBEDDING_MODEL=BAAI/bge-m3       # 模型名（local模式）/ API模型名（api模式）
      EMBEDDING_DIMENSIONS=1024         # 向量维度
      EMBEDDING_API_KEY=xxx             # API Key（仅 api 模式需要）
      EMBEDDING_BASE_URL=https://...    # API Base URL（仅 api 模式需要）
    """

    def __init__(self):
        self._backend: Optional[EmbeddingBackend] = None
        self._provider = getattr(settings, "EMBEDDING_PROVIDER", "auto")
        self._model_name = getattr(settings, "EMBEDDING_MODEL", "BAAI/bge-m3")
        self._initialized = False

    async def initialize(self):
        """初始化 Embedding 服务（自动选择最佳后端）"""
        provider = self._provider.lower()

        # ---- 分发到具体后端 ----
        if provider in ("local", "fastembed", "auto"):
            success = await self._try_init_fastembed()
            if not success and provider == "auto":
                logger.warning("⚠️ FastEmbed 初始化失败，尝试 SentenceTransformers...")
                success = await self._try_init_sentence_transformer()
            if not success and provider == "auto":
                logger.warning("⚠️ 本地模型均不可用，尝试 OpenAI API...")
                success = await self._try_init_openai_api()
            if not success:
                raise RuntimeError(
                    "所有 Embedding 后端初始化失败！请至少确保以下之一可用：\n"
                    "  1. pip install fastembed (推荐，本地 BGE-M3)\n"
                    "  2. pip install sentence-transformers\n"
                    "  3. 设置 EMBEDDING_API_KEY + EMBEDDING_BASE_URL (云 API)"
                )

        elif provider == "sentence_transformer":
            success = await self._try_init_sentence_transformer()
            if not success:
                raise RuntimeError("SentenceTransformers 初始化失败")

        elif provider in ("openai_api", "openai", "api"):
            success = await self._try_init_openai_api()
            if not success:
                raise RuntimeError("OpenAI Compatible API 初始化失败，请检查 EMBEDDING_API_KEY")

        else:
            raise ValueError(f"未知的 Embedding Provider: {provider}")

        self._initialized = True
        logger.info(
            f"✅ EmbeddingService 初始化成功 | "
            f"后端={self._backend.backend_name} | "
            f"维度={self.dimensions}"
        )

    async def _try_init_fastembed(self) -> bool:
        """尝试初始化 FastEmbed 后端"""
        try:
            self._backend = FastEmbedBackend(model_name=self._model_name)
            healthy = await self._backend.health_check()
            if healthy:
                logger.info(f"✅ FastEmbed 后端就绪: {self._model_name}")
                return True
            else:
                self._backend = None
                return False
        except ImportError as e:
            logger.info(f"FastEmbed 未安装 ({e})，跳过")
            return False
        except Exception as e:
            logger.warning(f"FastEmbed 初始化异常: {e}")
            self._backend = None
            return False

    async def _try_init_sentence_transformer(self) -> bool:
        """尝试初始化 SentenceTransformers 后端（备选，支持更多模型）"""
        try:
            # 如果当前模型在 FastEmbed 白名单中但 FastEmbed 失败，
            # 或模型不在白名单中（如 bge-m3 / bge-large-zh），用 ST 兜底
            model = self._model_name
            
            self._backend = SentenceTransformerBackend(model_name=model)
            healthy = await self._backend.health_check()
            if healthy:
                logger.info(f"✅ SentenceTransformer 后端就绪: {model} (FastEmbed 不可用时降级)")
                return True
            else:
                self._backend = None
                return False
        except ImportError as e:
            logger.info(f"sentence-transformers 未安装 ({e})，跳过 ST 后端")
            return False
        except Exception as e:
            logger.warning(f"SentenceTransformer 初始化异常: {e}")
            self._backend = None
            return False

    async def _try_init_openai_api(self) -> bool:
        """尝试初始化 OpenAI 兼容 API 后端"""
        api_key = getattr(settings, "EMBEDDING_API_KEY", "") or getattr(settings, "LLM_API_KEY", "")
        base_url = getattr(settings, "EMBEDDING_BASE_URL", "")
        model = self._model_name

        # 如果 model 名看起来像本地模型名（含斜杠），切换为默认 API 模型
        if "/" in model:
            model = "embedding-3"  # 默认用智谱

        if not api_key:
            logger.warning("未设置 EMBEDDING_API_KEY 或 LLM_API_KEY，无法使用 API 模式")
            return False

        try:
            self._backend = OpenAICompatibleBackend(
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
            healthy = await self._backend.health_check()
            if healthy:
                logger.info(f"✅ OpenAI API 后端就绪: {model}")
                return True
            else:
                self._backend = None
                return False
        except Exception as e:
            logger.warning(f"OpenAI API 初始化异常: {e}")
            self._backend = None
            return False

    # ===== 公共接口（代理到当前后端）=====

    @property
    def backend(self) -> EmbeddingBackend:
        assert self._backend is not None, "EmbeddingService 尚未初始化"
        return self._backend

    @property
    def dimensions(self) -> int:
        return self.backend.dimensions

    @property
    def provider_name(self) -> str:
        return self.backend.backend_name

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量向量化（文档入库用）"""
        return await self.backend.embed_texts(texts)

    async def embed_query(self, query: str) -> List[float]:
        """单条向量化（检索查询用）"""
        return await self.backend.embed_query(query)

    async def health_check(self) -> bool:
        return await self.backend.health_check()

    def get_info(self) -> Dict[str, Any]:
        """返回 Embedding 服务信息"""
        return {
            "provider": self.provider_name,
            "model": self._model_name,
            "dimensions": self.dimensions,
            "initialized": self._initialized,
        }


# ============================================================
# LangChain 兼容封装
# ============================================================

class LangChainEmbeddingsAdapter:
    """
    将我们的 EmbeddingService 适配为 LangChain 的 Embeddings 接口
    
    用法（替代之前的 OpenAIEmbeddings）：
    
        adapter = LangChainEmbeddingsAdapter(embedding_service)
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name="...",
            embedding=adapter,  # <-- 这里传入适配器
        )
    """

    def __init__(self, service: EmbeddingService):
        self._service = service

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain 接口：批量文档向量化"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # 如果已经在异步事件循环中，用 create_task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._service.embed_texts(texts))
                return future.result()
        else:
            return asyncio.run(self._service.embed_texts(texts))

    def embed_query(self, query: str) -> List[float]:
        """LangChain 接口：查询向量化"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._service.embed_query(query))
                return future.result()
        else:
            return asyncio.run(self._service.embed_query(query))

    @property
    def dimensions(self) -> int:
        return self._service.dimensions
