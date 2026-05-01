"""
重排序服务 — BGE-Reranker-V2-M3 via Jina AI API

Cross-encoder 重排序器，精度远超向量检索（bi-encoder）。
将 (query, candidate_text) 对作为输入，输出 0-1 的相关度分数。

API: Jina AI Reranker API
  - 模型: jina-reranker-v2-base-multilingual
  - 支持中文, 收费但价格低廉
"""

import logging
import time
from typing import List, Tuple, Optional
from datetime import datetime

import httpx

from app.config import settings
from app.models.enhanced import (
    ScoredChunk, SourceType, AuthorityLevel,
)

logger = logging.getLogger(__name__)


class LegalReranker:
    """
    法律专用重排序器

    输入: user_query + candidate_chunks (来自多路检索)
    输出: 重排序后的 chunks，含融合分

    融合公式:
      final_score = 0.4 × rerank_score
                  + 0.25 × authority_score
                  + 0.15 × timeliness_score
                  + 0.10 × vector_score
                  + 0.10 × bm25_score
    """

    # Cross-encoder 权重（向量检索给的初筛，Cross-encoder 做精排）
    RERANK_WEIGHT = 0.40
    AUTHORITY_WEIGHT = 0.25
    TIMELINESS_WEIGHT = 0.15
    VECTOR_WEIGHT = 0.10
    BM25_WEIGHT = 0.10

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.EMBEDDING_API_KEY
        self.model = model or "jina-reranker-v2-base-multilingual"
        self._client: Optional[httpx.AsyncClient] = None
        self._initialized = False

    async def initialize(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        self._initialized = True
        logger.info(f"✅ 重排序器就绪: model={self.model}")

    async def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int = 5,
    ) -> List[ScoredChunk]:
        """
        对候选结果进行重排序

        参数:
          query: 用户查询
          candidates: 多路检索的候选结果
          top_k: 最终返回条数

        返回:
          按融合分降序排列的 ScoredChunk 列表
        """
        if not candidates:
            return []

        # 1. 调用 Cross-encoder 获取精排分
        pairs = [(query, c.text) for c in candidates]
        rerank_scores = await self._call_reranker_api(pairs)

        # 2. 计算融合分
        now = datetime.now()
        for i, chunk in enumerate(candidates):
            chunk.rerank_score = rerank_scores[i] if i < len(rerank_scores) else 0.0
            chunk.authority_score = self._calc_authority(chunk)
            chunk.timeliness_score = self._calc_timeliness(chunk, now)

            chunk.final_score = (
                self.RERANK_WEIGHT * chunk.rerank_score
                + self.AUTHORITY_WEIGHT * chunk.authority_score
                + self.TIMELINESS_WEIGHT * chunk.timeliness_score
                + self.VECTOR_WEIGHT * chunk.vector_score
                + self.BM25_WEIGHT * chunk.bm25_score
            )

        # 3. 排序并截断
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates[:top_k]

    async def _call_reranker_api(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:
        """
        调用 Jina Reranker API

        API 文档: POST https://api.jina.ai/v1/rerank
        Request:
          {
            "model": "jina-reranker-v2-base-multilingual",
            "query": "...",
            "documents": ["...", "..."],
            "top_n": len(documents)
          }
        Response:
          {
            "results": [
              {"index": 0, "relevance_score": 0.95, "document": {"text": "..."}},
              ...
            ]
          }
        """
        if not pairs:
            return []

        query = pairs[0][0]
        documents = [p[1] for p in pairs]

        try:
            resp = await self._client.post(
                "https://api.jina.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": len(documents),
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # 按 index 映射回原始顺序
            results = sorted(data["results"], key=lambda r: r["index"])
            return [r["relevance_score"] for r in results]

        except Exception as e:
            logger.error(f"重排序 API 调用失败: {e}")
            # API 失败时回退到向量分
            return [c.vector_score for c in self._dummy_candidates(len(pairs))]

    # ===== 权重计算 =====

    def _calc_authority(self, chunk: ScoredChunk) -> float:
        """权威性评分（0~1）"""
        source_type = chunk.source_type
        if isinstance(source_type, str):
            try:
                source_type = SourceType(source_type)
            except ValueError:
                pass

        authority_map = {
            SourceType.GUIDE_CASE: 1.0,         # 指导案例
            SourceType.LAW: 0.95,                # 法律
            SourceType.JUDICIAL_INTERP: 0.90,    # 司法解释
            SourceType.REGULATION: 0.85,         # 行政法规
            SourceType.JUDGMENT: 0.70,           # 裁判文书
            SourceType.COMMENTARY: 0.60,         # 权威评注
            SourceType.LOCAL_LAW: 0.50,          # 地方法规
        }

        return authority_map.get(source_type, 0.30)

    def _calc_timeliness(self, chunk: ScoredChunk, now: datetime) -> float:
        """时效性评分——越新的法律越优先"""
        # 从 payload 中尝试提取公布日期
        publish_date = chunk.metadata.get("gbrq") or chunk.metadata.get("publish_date", "")

        if not publish_date:
            return 0.5  # 无日期，给中间分

        try:
            # 处理多种日期格式
            if isinstance(publish_date, str):
                if len(publish_date) == 10:  # "2023-12-29"
                    pub = datetime.strptime(publish_date, "%Y-%m-%d")
                elif len(publish_date) == 7:  # "2023-12"
                    pub = datetime.strptime(publish_date, "%Y-%m")
                elif len(publish_date) == 4:  # "2023"
                    pub = datetime.strptime(publish_date, "%Y")
                else:
                    return 0.5
            else:
                return 0.5

            years_ago = (now - pub).days / 365.0

            if years_ago <= 3:
                return 1.0                  # 近3年
            elif years_ago <= 8:
                return 1.0 - (years_ago - 3) * 0.08  # 3~8年线性衰减
            elif years_ago <= 15:
                return 0.6 - (years_ago - 8) * 0.04  # 8~15年继续衰减
            else:
                return 0.3                  # 超过15年的旧法

        except (ValueError, TypeError):
            return 0.5

    def _dummy_candidates(self, n: int) -> List[ScoredChunk]:
        """API 失败时的占位数据"""
        return [ScoredChunk(text="", metadata={}) for _ in range(n)]

    @property
    def is_ready(self) -> bool:
        return self._initialized

    async def shutdown(self):
        if self._client:
            await self._client.aclose()
        logger.info("LegalReranker 已关闭")
