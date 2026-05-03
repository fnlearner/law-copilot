"""
搜索接口 - 法条/案例快速检索
- LAWS/ECONOMIC → Qdrant 语义检索
- CASES → SQLite FTS5 BM25 全文检索
- ALL → 两者合并
"""

import time
import logging
from fastapi import APIRouter, Request, Query

from app.models.schemas import SearchRequest, SearchResult, SearchResponse, SearchScope
from app.services.rag_service import RAGService
from app.services.judgment_service import JudgmentSearchService
from app.config import settings
from app.utils.metadata import build_title

logger = logging.getLogger(__name__)
router = APIRouter()


def get_rag(request: Request) -> RAGService:
    return request.app.state.rag_service


def get_judgment(request: Request) -> JudgmentSearchService:
    return request.app.state.judgment_service


@router.post("/query", response_model=SearchResponse)
async def search_documents(
    body: SearchRequest,
    request: Request,
):
    """
    文档搜索接口
    CASES → SQLite BM25 全文检索
    LAWS / ECONOMIC → Qdrant 语义检索
    ALL → 两者合并
    """
    start = time.time()
    rag = get_rag(request)
    jdg = get_judgment(request)

    scope_map = {
        "SearchScope.ALL": "all",
        "SearchScope.LAWS": "law",
        "SearchScope.CASES": "case",
        "SearchScope.ECONOMIC": "economic",
    }
    search_scope = scope_map.get(str(body.scope), "all")

    try:
        all_results = []

        # === 裁判文书搜索（CASES / ALL）===
        if search_scope in ("case", "all") and jdg.is_ready:
            jdg_results = jdg.search(
                keywords=[body.query],  # 原始查询词直接搜
                limit=body.top_k,
            )
            for r in jdg_results:
                sr = SearchResult(**jdg.format_result(r))
                all_results.append(sr)

        # === 法条/经济搜索（LAW / ECONOMIC / ALL）===
        if search_scope in ("law", "economic", "all"):
            law_results = await rag.search(
                query=body.query,
                top_k=body.top_k,
                scope=search_scope if search_scope != "all" else "all",
                score_threshold=settings.SCORE_THRESHOLD,
            )
            for r in law_results:
                meta = r["metadata"]
                sr = SearchResult(
                    doc_type=meta.get("doc_type", "unknown"),
                    title=meta.get("law_name", meta.get("title", "未知")),
                    content=r["content"],
                    source=meta.get("source_file", ""),
                    relevance_score=r["relevance_score"],
                    metadata={k: v for k, v in meta.items() if k not in ["content"]},
                )
                all_results.append(sr)

        # 合并排序：按相关度降序
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        all_results = all_results[:body.top_k]

        latency = int((time.time() - start) * 1000)
        return SearchResponse(
            results=all_results,
            total=len(all_results),
            query=body.query,
            scope=search_scope,
            latency_ms=latency,
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        return SearchResponse(
            results=[],
            total=0,
            query=body.query,
            scope=search_scope,
            latency_ms=int((time.time() - start) * 1000),
        )


@router.get("/stats")
async def search_stats(request: Request):
    """检索统计信息"""
    rag = get_rag(request)
    jdg = get_judgment(request)
    info = await rag.get_collection_info()
    info["judgments_ready"] = jdg.is_ready
    return info


@router.get("/suggestions")
async def get_search_suggestions(
    q: str = Query("", min_length=0, max_length=50, description="输入前缀"),
    limit: int = Query(8, ge=1, le=20),
):
    """搜索建议（法律关键词 + 罪名自动补全）"""
    q = q.strip().lower()
    results = []

    # 1. 诉讼高频关键词
    law_keywords = [
        ("公司法", "《中华人民共和国公司法》"),
        ("股东责任", "股东出资、连带责任"),
        ("股权转让", "股权转让限制与程序"),
        ("合同效力", "合同成立与效力判断"),
        ("违约责任", "违约金与损害赔偿"),
        ("劳动争议", "劳动关系认定与解除"),
        ("民间借贷", "借贷关系与利息保护"),
        ("交通事故", "侵权责任与赔偿标准"),
        ("婚姻家庭", "离婚财产分割与抚养权"),
        ("房产纠纷", "房屋买卖合同与产权"),
        ("知识产权", "著作权、商标权、专利权"),
        ("行政处罚", "行政行为的合法性审查"),
    ]

    if q:
        for keyword, desc in law_keywords:
            if q in keyword or q in desc.lower():
                results.append({"keyword": desc, "type": "law"})
            if len(results) >= limit:
                break
    else:
        for keyword, desc in law_keywords[:limit]:
            results.append({"keyword": desc, "type": "law"})

    return {"suggestions": results[:limit]}
