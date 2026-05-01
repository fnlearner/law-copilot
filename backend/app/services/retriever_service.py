"""
多路检索器 — BM25 + 向量 + 结构化字段 三路并行检索

设计原则:
  1. 三路互不依赖，可并行执行
  2. 每路返回自己的分数，在重排序阶段融合
  3. 结构化字段匹配直接命中原子知识单元（subject/behavior/condition）
"""

import asyncio
import logging
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from app.services.collection_manager import CollectionManager
from app.services.embedding_service import EmbeddingService
from app.models.enhanced import (
    ScoredChunk, SourceType, COLLECTION_CONFIGS,
)

logger = logging.getLogger(__name__)


class MultiPathRetriever:
    """
    三路检索器

    路1: BM25 关键词检索 — 精确命中法条编号、关键词
    路2: 语义向量检索 — 理解意图，召回相似内容
    路3: 结构化字段匹配 — 通过 subject/behavior 等字段精确筛选
    """

    def __init__(
        self,
        collection_mgr: CollectionManager,
        embedding_svc: EmbeddingService,
    ):
        self.collections = collection_mgr
        self.embedding = embedding_svc
        self._initialized = False

    async def initialize(self):
        self._initialized = True
        logger.info("✅ MultiPathRetriever 就绪")

    async def retrieve(
        self,
        query: str,
        top_k_per_path: int = 20,
        scope: Optional[str] = None,
        collections: Optional[List[str]] = None,
    ) -> List[ScoredChunk]:
        """
        三路检索，去重合并

        参数:
          query: 用户问题
          top_k_per_path: 每路的候选数
          scope: 过滤范围 (all/laws/cases)
          collections: 搜索哪些集合（默认全部）

        返回:
          合并去重后的候选列表（尚未重排序）
        """
        targets = collections or list(COLLECTION_CONFIGS.keys())

        # 并行执行三路检索
        bm25_task = self._bm25_retrieve(query, targets, top_k_per_path, scope)
        vector_task = self._vector_retrieve(query, targets, top_k_per_path, scope)
        struct_task = self._struct_retrieve(query, targets, top_k_per_path, scope)

        bm25_results, vector_results, struct_results = await asyncio.gather(
            bm25_task, vector_task, struct_task,
        )

        # 合并去重（按 text hash 去重）
        seen = set()
        merged = []

        for chunk in bm25_results + vector_results + struct_results:
            text_hash = hash(chunk.text[:200])
            if text_hash not in seen:
                seen.add(text_hash)
                merged.append(chunk)

        logger.info(
            f"多路检索: BM25={len(bm25_results)}, 向量={len(vector_results)}, "
            f"结构={len(struct_results)}, 合并去重={len(merged)}"
        )

        return merged

    # ===== 路1: BM25 关键词检索 =====

    async def _bm25_retrieve(
        self,
        query: str,
        collections: List[str],
        top_k: int,
        scope: Optional[str],
    ) -> List[ScoredChunk]:
        """BM25 关键词检索"""
        results = []

        for coll in collections:
            filter_cond = self._build_filter(scope, coll)
            raw = self.collections.search_bm25(
                collection_name=coll,
                query=query,
                top_k=top_k // len(collections),
                filter_conditions=filter_cond,
            )
            for r in raw:
                chunk = ScoredChunk(
                    text=r["payload"].get("full_text", ""),
                    metadata=r["payload"],
                    bm25_score=r["score"],
                    collection=coll,
                    source_type=self._detect_source(coll),
                    point_id=str(r["id"]),
                )
                results.append(chunk)

        return results

    # ===== 路2: 语义向量检索 =====

    async def _vector_retrieve(
        self,
        query: str,
        collections: List[str],
        top_k: int,
        scope: Optional[str],
    ) -> List[ScoredChunk]:
        """语义向量检索"""
        # 查询向量化
        query_vec = await self.embedding.embed_query(query)
        if not query_vec:
            logger.warning("向量检索失败：查询向量化返回空")
            return []

        results = []
        for coll in collections:
            filter_cond = self._build_filter(scope, coll)
            raw = self.collections.search_vector(
                collection_name=coll,
                query_vector=query_vec,
                top_k=top_k // len(collections),
                filter_conditions=filter_cond,
            )
            for r in raw:
                chunk = ScoredChunk(
                    text=r["payload"].get("full_text", r["payload"].get("content", "")),
                    metadata=r["payload"],
                    vector_score=r["score"],
                    collection=coll,
                    source_type=self._detect_source(coll),
                    point_id=str(r["id"]),
                )
                results.append(chunk)

        return results

    # ===== 路3: 结构化字段匹配 =====

    async def _struct_retrieve(
        self,
        query: str,
        collections: List[str],
        top_k: int,
        scope: Optional[str],
    ) -> List[ScoredChunk]:
        """
        结构化字段匹配

        针对法规类数据，提取 query 中的 subject/behavior 关键词，
        在 Qdrant 的 payload 字段上做精确过滤。

        示例:
          query = "公司回购股份的条件"
          → 提取 subject="公司", behavior="回购"
          → 在 Qdrant 中过滤 subject=公司 AND behavior=回购
        """
        # 从 query 中提取法律结构关键词
        struct_terms = self._extract_struct_terms(query)
        if not struct_terms:
            return []

        results = []
        for coll in collections:
            if coll == "laws" and struct_terms:
                # 对法规集合做结构化过滤
                filter_cond = {}
                if struct_terms.get("subject"):
                    filter_cond["subject"] = struct_terms["subject"]
                if struct_terms.get("behavior"):
                    filter_cond["behavior"] = struct_terms["behavior"]

                # 先向量检索，再在过滤条件中匹配
                query_vec = await self.embedding.embed_query(query)
                if query_vec:
                    raw = self.collections.search_vector(
                        collection_name=coll,
                        query_vector=query_vec,
                        top_k=top_k,
                        filter_conditions=filter_cond,
                    )
                    for r in raw:
                        chunk = ScoredChunk(
                            text=r["payload"].get("full_text", ""),
                            metadata=r["payload"],
                            vector_score=r["score"],
                            struct_score=0.8,  # 结构化匹配加分
                            collection=coll,
                            source_type=SourceType.LAW,
                            point_id=str(r["id"]),
                        )
                        results.append(chunk)

        return results

    # ===== 工具方法 =====

    def _build_filter(
        self,
        scope: Optional[str],
        collection: str,
    ) -> Optional[Dict[str, Any]]:
        """构建 Qdrant 过滤条件"""
        if not scope or scope == "all":
            return None

        # scope 到 source_type 的映射
        scope_map = {
            "laws": SourceType.LAW.value,
            "cases": SourceType.JUDGMENT.value,
            "economic": SourceType.LAW.value,  # 经济类 -> 法规集合
        }
        mapped = scope_map.get(scope)
        if mapped:
            return {"source_type": mapped}
        return None

    def _detect_source(self, collection: str) -> SourceType:
        """根据集合名推断来源类型"""
        source_map = {
            "laws": SourceType.LAW,
            "judgments": SourceType.JUDGMENT,
            "commentaries": SourceType.COMMENTARY,
        }
        return source_map.get(collection, SourceType.LAW)

    def _extract_struct_terms(self, query: str) -> Dict[str, str]:
        """
        从查询中提取法律结构关键词

        简单实现: 通过关键词表匹配
        增强版: 调用 NER 模型
        """
        # 常见法律主体
        subjects = ["公司", "股东", "董事", "监事", "合伙人", "债权人",
                     "债务人", "劳动者", "用人单位", "消费者", "经营者"]

        # 常见法律行为
        behaviors = ["出资", "股权转让", "回购", "清算", "解散",
                      "违约责任", "侵权", "抵押", "担保", "诉讼",
                      "仲裁", "执行", "破产", "重整"]

        result = {}
        query_lower = query

        for s in subjects:
            if s in query_lower:
                result["subject"] = s
                break

        for b in behaviors:
            if b in query_lower:
                result["behavior"] = b
                break

        return result

    def extract_article_refs(self, query: str) -> List[str]:
        """从查询中提取法条引用（如"民法典第585条"）"""
        pattern = r'(《[^》]+》)第([^条]+)条'
        matches = re.findall(pattern, query)
        return [f"{m[0]}第{m[1]}条" for m in matches]

    @property
    def is_ready(self) -> bool:
        return self._initialized
