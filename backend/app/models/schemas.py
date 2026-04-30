"""
LawCopilot 数据模型定义
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ===== 枚举 =====

class DocumentType(str, Enum):
    """文档类型"""
    LAW = "law"              # 法律法规
    CASE = "case"            # 判例案例
    AGREEMENT = "agreement"  # 合同模板
    JUDICIAL = "judicial"    # 司法解释
    OTHER = "other"


class SearchScope(str, Enum):
    """检索范围"""
    ALL = "all"
    LAWS = "laws"            # 仅法律法规
    CASES = "cases"          # 仅判例
    ECONOMIC = "economic"    # 经济类（核心）


class TaskType(str, Enum):
    """任务类型"""
    SEARCH_LAW = "search_law"           # 法条检索
    ANALYZE_CASE = "analyze_case"       # 案例分析
    GENERATE_DOC = "generate_doc"       # 文书生成
    LEGAL_RESEARCH = "legal_research"   # 法律研究
    CHAT = "chat"                       # 通用对话


# ===== 请求模型 =====

class ChatRequest(BaseModel):
    """对话请求"""
    message: str = Field(..., min_length=1, max_length=4000, description="用户提问")
    session_id: Optional[str] = Field(None, description="会话ID，为空则新建")
    task_type: TaskType = Field(TaskType.LEGAL_RESEARCH, description="任务类型")
    scope: SearchScope = Field(SearchScope.ECONOMIC, description="检索范围")
    top_k: int = Field(5, ge=1, le=20, description="检索返回条数")

    @field_validator("scope", mode="before")
    @classmethod
    def normalize_scope(cls, v):
        """兼容前端 'searchScope.ALL' 格式"""
        if isinstance(v, str):
            if "." in v:
                v = v.split(".")[-1].lower()
            mapping = {
                "all": SearchScope.ALL,
                "laws": SearchScope.LAWS,
                "cases": SearchScope.CASES,
                "economic": SearchScope.ECONOMIC,
            }
            return mapping.get(v.lower(), SearchScope.ECONOMIC)
        return v


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., min_length=1, max_length=2000, description="搜索关键词")
    scope: SearchScope = Field(SearchScope.ALL)
    top_k: int = Field(5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")

    @field_validator("scope", mode="before")
    @classmethod
    def normalize_scope(cls, v):
        """兼容前端 'searchScope.ALL' 格式和直接字符串值"""
        if isinstance(v, str):
            if "." in v:
                v = v.split(".")[-1].lower()
            mapping = {
                "all": SearchScope.ALL,
                "laws": SearchScope.LAWS,
                "cases": SearchScope.CASES,
                "economic": SearchScope.ECONOMIC,
            }
            return mapping.get(v.lower(), SearchScope.ALL)
        return v


class DocumentUploadRequest(BaseModel):
    """文档上传元信息"""
    title: str
    doc_type: DocumentType
    category: Optional[str] = None
    tags: List[str] = []
    description: Optional[str] = None


# ===== 响应模型 =====

class SourceReference(BaseModel):
    """引用来源"""
    doc_type: str                    # 文档类型
    title: str                       # 标题
    source: str                      # 来源/出处
    content_snippet: str             # 内容片段
    relevance_score: float           # 相关度得分
    article_number: Optional[str] = None  # 条文编号


class ChatResponse(BaseModel):
    """对话响应"""
    reply: str                       # 回复内容
    references: List[SourceReference] = []  # 引用来源列表
    session_id: str                  # 会话ID
    task_type: str                   # 任务类型
    model_used: str                  # 使用模型
    latency_ms: int                  # 耗时(ms)


class SearchResult(BaseModel):
    """搜索结果项"""
    doc_type: str
    title: str
    content: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    """搜索响应"""
    results: List[SearchResult]
    total: int
    query: str
    scope: str
    latency_ms: int


class DocumentInfo(BaseModel):
    """文档信息"""
    id: str
    title: str
    doc_type: str
    category: Optional[str]
    tags: List[str]
    status: str  # processing / ready / error
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class APIResponse(BaseModel):
    """通用API响应"""
    code: int = 0
    message: str = "success"
    data: Any = None
