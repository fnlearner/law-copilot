"""
搜索接口 - 法条/案例快速检索
"""

import time
import logging
from fastapi import APIRouter, Request, Query
from typing import Optional

from app.models.schemas import SearchRequest, SearchResult, SearchResponse, SearchScope
from app.services.rag_service import RAGService
from app.config import settings


logger = logging.getLogger(__name__)
router = APIRouter()


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


@router.post("/query", response_model=SearchResponse)
async def search_documents(
    body: SearchRequest,
    request: Request,
):
    """
    文档搜索接口
    支持按范围过滤（法律/案例/全部）
    返回带相关度评分的搜索结果
    """
    start = time.time()
    rag = request.app.state.rag_service

    scope_map = {
        "searchScope.ALL": "all",
        "searchScope.LAWS": "law",
        "searchScope.CASES": "case",
        "searchScope.ECONOMIC": "economic",
    }
    search_scope = scope_map.get(str(body.scope), "all")

    try:
        raw_results = await rag.search(
            query=body.query,
            top_k=body.top_k,
            scope=search_scope,
            score_threshold=settings.SCORE_THRESHOLD,
        )

        results = []
        for r in raw_results:
            meta = r["metadata"]
            sr = SearchResult(
                doc_type=meta.get("doc_type", "unknown"),
                title=meta.get("file_name", meta.get("title", "未知")),
                content=r["content"],
                source=meta.get("source_file", ""),
                relevance_score=r["relevance_score"],
                metadata={k: v for k, v in meta.items() if k not in ["content"]},
            )
            results.append(sr)

        latency = int((time.time() - start) * 1000)

        return SearchResponse(
            results=results,
            total=len(results),
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


@router.get("/suggestions")
async def get_search_suggestions(
    q: str = Query(..., min_length=1, description="输入前缀"),
    limit: int = Query(8, ge=1, le=20),
):
    """搜索建议（基于常见法律关键词）"""

    # 经济类法律高频关键词库
    economic_keywords = [
        # 公司法
        ("公司法", "《中华人民共和国公司法》"),
        ("股东责任", "股东出资、连带责任"),
        ("股权转让", "股权转让限制与程序"),
        ("公司解散", "公司清算与解散"),
        ("董监高责任", "董事监事高管义务"),
        # 合同
        ("合同效力", "合同成立与效力判断"),
        ("违约责任", "违约金与损害赔偿"),
        ("格式条款", "格式合同规制"),
        ("解除合同", "合同解除条件与后果"),
        # 证券金融
        ("内幕交易", "内幕信息与交易责任"),
        ("虚假陈述", "证券虚假陈述赔偿"),
        ("信息披露", "上市公司披露义务"),
        ("操纵市场", "市场操纵认定标准"),
        # 破产
        ("破产重整", "企业破产重整程序"),
        ("破产债权", "债权申报与清偿顺序"),
        ("管理人制度", "破产管理人职责"),
        # 劳动法
        ("竞业限制", "竞业禁止协议效力"),
        ("经济性裁员", "企业裁员法定程序"),
        ("股权激励", "员工股权激励方案"),
        ("劳动合同", "劳动关系认定"),
        # 知识产权
        ("商业秘密", "商业秘密保护范围"),
        ("专利侵权", "专利侵权判定标准"),
        ("商标侵权", "商标侵权赔偿计算"),
        # 反垄断
        ("垄断协议", "横向/纵向垄断协议"),
        ("经营者集中", "反垄断审查申报"),
        ("滥用支配地位", "市场支配地位滥用"),
        # 涉外
        ("涉外仲裁", "国际商事仲裁执行"),
        ("管辖权异议", "涉外民商事管辖"),
    ]

    suggestions = [item[1] for item in economic_keywords if q in item[0]]
    return {"suggestions": suggestions[:limit]}


@router.get("/stats")
async def search_stats(request: Request):
    """检索统计信息"""
    rag = request.app.state.rag_service
    info = await rag.get_collection_info()
    return info
