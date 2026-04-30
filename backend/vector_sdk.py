"""
vector_sdk.py — Minimal glue layer for law-copilot vector operations.

职责边界（只做这三件事）：
  1. 封装 embedding API（Jina / OpenAI compatible）调用和批量处理
  2. 封装 Qdrant 的写入 / 检索 / 删除
  3. 可选：接入 ReRank（bge-reranker / Cohere）

不包含：
  - 文档解析（你们已有 split_articles）
  - 文件 I/O
  - 业务 metadata 推断（doc_type 等，保留在调用方）
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import requests

# ─────────────────────────────────────────────────────────────────
# 1. Embedding Provider Protocol
# ─────────────────────────────────────────────────────────────────

class EmbeddingProvider(Protocol):
    """Embedding provider 接口。任何实现了 embed() 的对象都可以注入。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class JinaEmbeddingProvider:
    """
    Jina AI embedding API（OpenAI compatible 端点）。
    环境变量：JINA_API_KEY 或 EMBEDDING_API_KEY
    """
    model: str = "jina-embeddings-v3"
    api_key: str | None = None
    batch_size: int = 32
    timeout: int = 60
    _base_url: str = "https://api.jina.ai/v1/embeddings"

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("JINA_API_KEY") or os.environ.get("EMBEDDING_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY / EMBEDDING_API_KEY environment variable not set")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化，支持长文本自动截断（Jina v3 支持 up to 8192 tokens）。"""
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            payload = {"model": self.model, "input": batch}
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            resp = requests.post(self._base_url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()["data"]
            # 按 index 排序保证顺序
            data.sort(key=lambda x: x["index"])
            results.extend(item["embedding"] for item in data)
        return results

    def embed_single(self, text: str) -> list[float]:
        """单条向量化（内部批量也走 embed，只是方便外部调用）。"""
        return self.embed([text])[0]


# ─────────────────────────────────────────────────────────────────
# 2. Vector Store（Qdrant）
# ─────────────────────────────────────────────────────────────────

@dataclass
class VectorPoint:
    id: int | str  # Qdrant accepts int or UUID string
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class SearchResult:
    id: int | str
    score: float
    payload: dict[str, Any]


@dataclass
class QdrantStore:
    """
    Qdrant 向量存储封装。

    用法：
        store = QdrantStore(collection="law_copilot_laws", host="localhost", port=6333)
        store.upsert(points)
        results = store.search(query_vector, top_k=5, filter={"doc_type": "civil"})
    """
    collection: str
    host: str = "localhost"
    port: int = 6333
    vector_size: int = 1024  # jina-embeddings-v3 = 1024
    distance: str = "Cosine"
    timeout: int = 30
    _client: Any = field(default=None, repr=False)

    @property
    def client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(host=self.host, port=self.port, timeout=self.timeout)
        return self._client

    def _ensure_collection(self) -> None:
        """确保 collection 存在，不存在则自动创建。"""
        from qdrant_client.models import Distance, VectorParams
        cols = [c.name for c in self.client.get_collections().collections]
        if self.collection not in cols:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance[self.distance],
                ),
            )

    # ── 写入 ─────────────────────────────────────────────────────

    def upsert(self, points: list[VectorPoint], batch_size: int = 100) -> int:
        """
        批量写入向量点。
        返回成功写入的点数。
        """
        self._ensure_collection()
        from qdrant_client.models import PointStruct
        structs = [
            PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ]
        for i in range(0, len(structs), batch_size):
            batch = structs[i : i + batch_size]
            self.client.upsert(collection_name=self.collection, points=batch)
        return len(structs)

    def delete(self, point_ids: list[int | str]) -> None:
        """根据 ID 删除向量点。"""
        from qdrant_client.models import PointIdsList
        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=point_ids),
        )

    # ── 检索 ─────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        向量相似度检索。
        filter：Qdrant filter 格式，如 {"key": "doc_type", "match": {"value": "civil"}}
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        search_kwargs: dict[str, Any] = {
            "collection_name": self.collection,
            "query_vector": query_vector,
            "limit": top_k,
        }
        if filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter.items()
            ]
            search_kwargs["query_filter"] = Filter(must=conditions)
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        results = self.client.search(**search_kwargs)
        return [
            SearchResult(id=r.id, score=r.score, payload=r.payload or {})
            for r in results
        ]

    # ── 辅助 ─────────────────────────────────────────────────────

    def count(self) -> int:
        """返回 collection 中的总点数。"""
        return self.client.get_collection(collection_name=self.collection).points_count

    def scroll(
        self,
        filter: dict[str, Any] | None = None,
        limit: int = 1000,
        offset: str | None = None,
    ) -> tuple[list[SearchResult], str | None]:
        """遍历所有点，yield (results, next_offset)。"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        kwargs: dict[str, Any] = {
            "collection_name": self.collection,
            "with_vectors": False,
            "with_payload": True,
            "limit": limit,
        }
        if filter:
            kwargs["scroll_filter"] = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter.items()]
            )
        if offset:
            kwargs["offset"] = offset

        results, next_offset = self.client.scroll(**kwargs)
        points = [
            SearchResult(id=str(pt.id), score=0.0, payload=pt.payload or {})
            for pt in results
        ]
        return points, next_offset or None


# ─────────────────────────────────────────────────────────────────
# 3. ReRanker（可选，现阶段可跳过）
# ─────────────────────────────────────────────────────────────────

@dataclass
class ReRankerResult:
    index: int          # 原始列表中的下标
    document: str       # chunk 原文
    score: float        # 原始语义相似度
    rerank_score: float # 重排后的相关性分数
    payload: dict[str, Any]


class CohereReRanker:
    """
    Cohere ReRank（https://cohere.com/rerank）。
    环境变量：COHERE_API_KEY

    用法：
        reranker = CohereReRanker(top_n=5)
        reranked = reranker.rerank(query, chunks, base_scores)
    """
    model: str = "rerank-multilingual-v3.0"
    top_n: int = 5
    api_key: str | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY environment variable not set")

    def rerank(
        self,
        query: str,
        documents: list[str],
        base_scores: list[float] | None = None,
    ) -> list[ReRankerResult]:
        import cohere
        co = cohere.Client(self.api_key)
        response = co.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=self.top_n,
            return_documents=False,
        )
        base_scores = base_scores or [1.0] * len(documents)
        results: list[ReRankerResult] = []
        seen: set[int] = set()
        for item in response.results:
            results.append(ReRankerResult(
                index=item.index,
                document=documents[item.index],
                score=base_scores[item.index],
                rerank_score=item.relevance_score,
                payload={},
            ))
            seen.add(item.index)
        # 没进 top_n 的文档也保留（rerank_score = 0）
        for i, doc in enumerate(documents):
            if i not in seen:
                results.append(ReRankerResult(
                    index=i, document=doc,
                    score=base_scores[i], rerank_score=0.0, payload={}
                ))
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results


# ─────────────────────────────────────────────────────────────────
# 4. LawVectorSDK — 整合以上三个组件
# ─────────────────────────────────────────────────────────────────

@dataclass
class LawChunk:
    """法律文档 chunk。分块逻辑由调用方负责，这里只存结果。"""
    content: str
    chunk_id: str  # 稳定字符串 ID（用于业务层去重）
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_point_id(self) -> int:
        """把 chunk_id 映射为 Qdrant 可接受的整数 ID（确定性，不受 Python hash 随机化影响）。"""
        digest = hashlib.sha256(self.chunk_id.encode()).digest()
        return int.from_bytes(digest[:4], byteorder="big")  # 取前4字节=32bit整数

    def as_point_id_str(self) -> str:
        """返回字符串形式的 point ID（用于 delete 等需要 string 的场景）。"""
        return str(self.as_point_id())


@dataclass
class LawQueryResult:
    """查询结果，包含原始 chunk 信息和得分。"""
    chunk_id: str
    content: str
    score: float
    rerank_score: float | None
    metadata: dict[str, Any]


class LawVectorSDK:
    """
    Minimal vector SDK for law-copilot。

    典型用法：
        sdk = LawVectorSDK()
        sdk.ingest(chunks=[LawChunk(...)])        # 向量化 + 写入 Qdrant
        results = sdk.query("合同违约怎么处理")    # 检索 + 可选 ReRank
    """

    def __init__(
        self,
        collection: str = "law_copilot_laws",
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: QdrantStore | None = None,
        reranker: CohereReRanker | None = None,
        host: str = "localhost",
        port: int = 6333,
    ):
        self.embedding = embedding_provider or JinaEmbeddingProvider()
        self.store = vector_store or QdrantStore(collection=collection, host=host, port=port)
        self.reranker = reranker

    def ingest(self, chunks: list[LawChunk]) -> int:
        """向量化 chunks 并写入 Qdrant。返回写入数量。"""
        texts = [c.content for c in chunks]
        vectors = self.embedding.embed(texts)

        points = [
            VectorPoint(
                id=c.as_point_id(),   # Qdrant 要求整数或 UUID，用 hash 映射
                vector=vec,
                payload={
                    "content": c.content,
                    "chunk_id": c.chunk_id,   # 存业务层 ID，方便查询时还原
                    "metadata": c.metadata,
                },
            )
            for c, vec in zip(chunks, vectors)
        ]
        return self.store.upsert(points)

    def query(
        self,
        text: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        use_rerank: bool = False,
    ) -> list[LawQueryResult]:
        """
        查询法律文档。

        流程：
          1. 向量化 query
          2. Qdrant 向量检索（top_k * rerank_ratio，如果启用 ReRank）
          3. 可选：ReRank 重排
          4. 返回最终结果
        """
        top_k_search = top_k * 3 if (use_rerank and self.reranker) else top_k
        qvec = self.embedding.embed_single(text)
        hits = self.store.search(
            query_vector=qvec,
            top_k=top_k_search,
            filter=filter,
        )

        if use_rerank and self.reranker:
            docs = [h.payload.get("content", "") for h in hits]
            scores = [h.score for h in hits]
            reranked = self.reranker.rerank(text, docs, scores)
            return [
                LawQueryResult(
                    chunk_id=hits[r.index].payload.get("chunk_id", str(hits[r.index].id)),
                    content=r.document,
                    score=r.score,
                    rerank_score=r.rerank_score,
                    metadata=hits[r.index].payload.get("metadata", {}),
                )
                for r in reranked[:top_k]
            ]

        return [
            LawQueryResult(
                chunk_id=h.payload.get("chunk_id", str(h.id)),
                content=h.payload.get("content", ""),
                score=h.score,
                rerank_score=None,
                metadata=h.payload.get("metadata", {}),
            )
            for h in hits[:top_k]
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        """根据 chunk_id 列表删除向量点（内部用 SHA256 映射为确定性整数 ID）。"""
        def to_point_id(cid: str) -> int:
            digest = hashlib.sha256(cid.encode()).digest()
            return int.from_bytes(digest[:4], byteorder="big")

        point_ids = [to_point_id(cid) for cid in chunk_ids]
        self.store.delete(point_ids)

    def count(self) -> int:
        return self.store.count()
