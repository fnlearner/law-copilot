"""
LawCopilot - 律师助手
面向执业律师的经济类法律研究 Agent
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.routers import chat, search, document
from app.services.rag_service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化RAG服务
    app.state.rag_service = RAGService()
    await app.state.rag_service.initialize()
    yield
    # 关闭时清理资源
    await app.state.rag_service.shutdown()


app = FastAPI(
    title="LawCopilot API",
    description="面向执业律师的经济类法律研究 Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api/v1/chat", tags=["对话"])
app.include_router(search.router, prefix="/api/v1/search", tags=["检索"])
app.include_router(document.router, prefix="/api/v1/document", tags=["文档管理"])


@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "service": "law-copilot"}
