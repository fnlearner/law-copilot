"""
对话接口 - 律师提问 → RAG检索 + LLM生成专业法律分析
- CASES 范围 → 裁判文书 BM25 检索（SQLite FTS5）
- 其他范围 → 法条语义检索（Qdrant）
- 支持多轮对话上下文（基于 session_id 的内存存储）
"""

import time
import uuid
import logging
from collections import defaultdict
from fastapi import APIRouter, Request, Depends
from typing import Dict, Any, List

from app.models.schemas import (
    ChatRequest, ChatResponse, SourceReference, TaskType,
)
from app.services.rag_service import RAGService
from app.services.judgment_service import JudgmentSearchService
from app.config import settings
from app.utils.metadata import build_title

logger = logging.getLogger(__name__)
router = APIRouter()

# 内存会话存储（session_id -> {"history": [...], "key_facts": [...]}）
# key_facts: 每轮提取的法律关键信息列表（不过期，累积）
_session_store: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"history": [], "key_facts": []})


def get_rag(request: Request) -> RAGService:
    return request.app.state.rag_service


def get_judgment(request: Request) -> JudgmentSearchService:
    return request.app.state.judgment_service


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    body: ChatRequest,
    request: Request,
):
    """
    核心对话接口
    律师输入法律问题，系统检索相关法条/案例后交给LLM
    """
    start = time.time()
    rag = get_rag(request)
    jdg = get_judgment(request)

    session_id = body.session_id or str(uuid.uuid4())

    # 加载会话历史 & 已提取的关键事实
    session_data = _session_store.get(session_id, {"history": [], "key_facts": []})
    chat_history = session_data["history"]
    key_facts = session_data["key_facts"]

    scope_map = {
        "searchScope.ALL": "all",
        "searchScope.LAWS": "law",
        "searchScope.CASES": "case",
        "searchScope.ECONOMIC": "economic",
    }
    search_scope = scope_map.get(str(body.scope), "all")

    try:
        context_chunks = []

        # === 裁判文书检索（CASES / ALL）===
        if search_scope in ("case", "all") and jdg.is_ready:
            jdg_results = jdg.search(
                keywords=[body.message],
                limit=body.top_k,
            )
            for r in jdg_results:
                fmt = jdg.format_result(r)
                context_chunks.append({
                    "content": fmt["content"],
                    "relevance_score": fmt["relevance_score"],
                    "metadata": fmt["metadata"],
                    "source": "cail2018",
                })

        # === 法条检索（LAW / ECONOMIC / ALL）===
        if search_scope in ("law", "economic", "all"):
            law_chunks = await rag.search(
                query=body.message,
                top_k=body.top_k,
                scope=search_scope if search_scope != "all" else "all",
                score_threshold=settings.SCORE_THRESHOLD,
            )
            context_chunks.extend(law_chunks)

        # 合并排序
        context_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        context_chunks = context_chunks[:body.top_k]

        # Step 2: LLM 生成回答（传入历史上下文 + 已提取关键事实）
        reply = await rag.generate(
            question=body.message,
            context_chunks=context_chunks,
            task_type=str(body.task_type),
            chat_history=chat_history,
            key_facts=key_facts,
        )

        # Step 3: 构建引用来源
        references = []
        for chunk in context_chunks:
            meta = chunk.get("metadata", {})
            ref = SourceReference(
                doc_type=meta.get("doc_type", "unknown"),
                title=build_title(meta),
                source=chunk.get("source", meta.get("source_file", "")),
                content_snippet=chunk["content"][:300] + ("..." if len(chunk["content"]) > 300 else ""),
                relevance_score=chunk["relevance_score"],
                article_number=meta.get("article_number"),
            )
            references.append(ref)

        # 写入会话历史（最多保留最近 20 轮）
        _session_store[session_id]["history"].append({"role": "human", "content": body.message})
        _session_store[session_id]["history"].append({"role": "ai", "content": reply})
        if len(_session_store[session_id]["history"]) > 40:
            _session_store[session_id]["history"] = _session_store[session_id]["history"][-40:]

        # 提取本轮关键法律事实（异步，不阻塞回复）
        await rag.extract_key_facts(
            human_msg=body.message,
            ai_msg=reply,
            key_facts_list=_session_store[session_id]["key_facts"],
        )

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


@router.get("/session/{session_id}/debug")
async def debug_session(session_id: str):
    """调试端点：查看当前 session 的历史和关键事实"""
    data = _session_store.get(session_id, {"history": [], "key_facts": []})
    return {
        "session_id": session_id,
        "history_count": len(data["history"]),
        "key_facts": data["key_facts"],
    }


@router.post("/stream")
async def stream_chat(body: ChatRequest, request: Request):
    """
    流式对话接口（可选）
    """
    from fastapi.responses import StreamingResponse
    import json

    rag = get_rag(request)
    jdg = get_judgment(request)

    async def generate():
        try:
            context_chunks = []
            if jdg.is_ready:
                jdg_results = jdg.search(keywords=[body.message], limit=body.top_k)
                for r in jdg_results:
                    fmt = jdg.format_result(r)
                    context_chunks.append(fmt)

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
