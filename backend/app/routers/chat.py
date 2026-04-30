"""
对话接口 - 律师提问 → RAG检索 + LLM生成专业法律分析
"""

import time
import uuid
import logging
from fastapi import APIRouter, Request, Depends
from typing import Dict, Any

from app.models.schemas import (
    ChatRequest, ChatResponse, SourceReference, TaskType,
)
from app.services.rag_service import RAGService
from app.config import settings


logger = logging.getLogger(__name__)
router = APIRouter()


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    body: ChatRequest,
    rag: RAGService = Depends(get_rag_service),
):
    """
    核心对话接口

    律师输入法律问题，系统：
    1. 对问题进行语义检索，找到相关法条/案例
    2. 将问题+检索结果一起交给LLM
    3. 返回结构化的法律分析 + 引用来源
    """
    start = time.time()

    # 生成/复用会话ID
    session_id = body.session_id or str(uuid.uuid4())

    try:
        # Step 1: RAG 检索
        scope_map = {
            "searchScope.ALL": "all",
            "searchScope.LAWS": "law",
            "searchScope.CASES": "case",
            "searchScope.ECONOMIC": "economic",  # 默认经济类优先
        }
        search_scope = scope_map.get(str(body.scope), "all")

        context_chunks = await rag.search(
            query=body.message,
            top_k=body.top_k,
            scope=search_scope,
            score_threshold=settings.SCORE_THRESHOLD,
        )

        # Step 2: LLM 生成回答
        reply = await rag.generate(
            question=body.message,
            context_chunks=context_chunks,
            task_type=str(body.task_type),
        )

        # Step 3: 构建引用来源列表
        references = []
        for chunk in context_chunks:
            meta = chunk.get("metadata", {})
            ref = SourceReference(
                doc_type=meta.get("doc_type", "unknown"),
                title=meta.get("file_name", meta.get("title", "未知")),
                source=meta.get("source_file", ""),
                content_snippet=chunk["content"][:300] + ("..." if len(chunk["content"]) > 300 else ""),
                relevance_score=chunk["relevance_score"],
                article_number=meta.get("article_number"),
            )
            references.append(ref)

        latency = int((time.time() - start) * 1000)

        return ChatResponse(
            reply=reply,
            references=references,
            session_id=session_id,
            task_type=str(body.task_type),
            model_used=settings.LLM_MODEL,
            latency_ms=latency,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        latency = int((time.time() - start) * 1000)
        return ChatResponse(
            reply=f"抱歉，处理您的请求时出现错误：{str(e)}。请稍后重试或联系管理员。",
            references=[],
            session_id=session_id,
            task_type=str(body.task_type),
            model_used="error",
            latency_ms=latency,
        )


@router.post("/stream")
async def stream_chat(body: ChatRequest, request: Request):
    """
    流式对话接口（可选）
    适用于长文本生成的场景
    """
    from fastapi.responses import StreamingResponse
    import json

    rag = request.app.state.rag_service

    async def generate():
        try:
            context_chunks = await rag.search(query=body.message, top_k=body.top_k)

            # 流式调用LLM（简化实现）
            reply = await rag.generate(
                question=body.message,
                context_chunks=context_chunks,
                task_type=str(body.task_type),
            )

            yield f"data: {json.dumps({'type': 'content', 'text': reply}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
