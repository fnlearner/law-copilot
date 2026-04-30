"""
RAG 服务 - LawCopilot 核心检索增强生成引擎

完整能力：
  1. 文档处理 Pipeline（清洗 → 结构解析 → 法条级分块 → 元数据注入）
  2. 向量化存储（Qdrant）
  3. 混合语义检索
  4. LLM 上下文生成
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document as LangChainDocument
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

from app.config import settings
from app.services.document_pipeline import (
    DocumentProcessingPipeline,
    LegalChunk,
    LegalTextCleaner,
)
from app.services.embedding_service import (
    EmbeddingService,
    LangChainEmbeddingsAdapter,
)


logger = logging.getLogger(__name__)


# ===== 系统提示词（法律研究专用）=====
LAW_SYSTEM_PROMPT = """你是LawCopilot，一位专业的法律研究助手。你的用户是执业律师。

## 核心原则
1. **严谨准确**：所有法条引用必须注明具体条文编号和出处，不可凭记忆编造
2. **有据可依**：每个结论必须引用相关法律法规或判例作为支撑
3. **经济类专精**：你特别擅长以下领域：
   - 公司法、合同法（民法典合同编）、证券法
   - 破产法、反垄断法、税法
   - 金融监管法规（银行、保险、信托）
   - 知识产权（商业秘密、专利、商标）
   - 劳动法（经济性裁员、竞业限制、股权激励）
   - 涉外商事仲裁与诉讼

## 回答格式要求
- 开头先给出**结论摘要**（3-5句话）
- 然后分点详细论述，每点附**法条依据**
- 最后给出**实务建议**
- 引用格式：《法律名称》第X条[款] 或 (案号) 法院判决

## 重要提醒
- 如果问题涉及的法律规定存在争议或不同解释，请说明不同观点及理由
- 如果信息不足无法给出确定回答，请明确告知需要补充什么信息
- 不要提供具体的诉讼策略建议，而是分析法律风险和可能结果
"""


class RAGService:
    """RAG 核心服务 — 完整版（含真实 Embedding 模型）"""

    def __init__(self):
        self.qdrant_client: Optional[QdrantClient] = None
        self.vector_store: Optional[QdrantVectorStore] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.embeddings_adapter = None  # LangChain 兼容适配器
        self.llm = None
        self.pipeline: Optional[DocumentProcessingPipeline] = None
        self._initialized = False

    async def initialize(self):
        """初始化 RAG 服务"""
        try:
            # ════════════════════════════════════════
            # Step 1: 初始化 Embedding 服务（⭐ 核心！不是 LLM）
            # ════════════════════════════════════════
            logger.info("🔧 正在初始化 Embedding 模型...")
            
            self.embedding_service = EmbeddingService()
            await self.embedding_service.initialize()

            # 创建 LangChain 兼容的 Embeddings 对象
            self.embeddings_adapter = LangChainEmbeddingsAdapter(self.embedding_service)
            embedding_dim = self.embedding_service.dimensions
            embedding_info = self.embedding_service.get_info()
            logger.info(f"✅ Embedding 就绪: {embedding_info}")

            # ════════════════════════════════════════
            # Step 2: 初始化 LLM（生成式模型，用于回答）
            # ════════════════════════════════════════
            from langchain_openai import ChatOpenAI
            
            self.llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                openai_api_key=settings.LLM_API_KEY,
                openai_api_base=settings.LLM_BASE_URL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            logger.info("✅ LLM (DeepSeek) 初始化完成")

            # ════════════════════════════════════════
            # Step 3: 连接 Qdrant 向量数据库
            # ════════════════════════════════════════
            self.qdrant_client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
            )

            # 用 Embedding 的实际维度来创建 Collection（⚠️ 关键！）
            collections = [c.name for c in self.qdrant_client.get_collections().collections]
            if settings.QDRANT_COLLECTION not in collections:
                self.qdrant_client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=embedding_dim,  # ← 使用 Embedding 模型的实际维度，不再是硬编码
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    f"✅ 创建 Collection: {settings.QDRANT_COLLECTION} "
                    f"(dim={embedding_dim}, model={embedding_info['model']})"
                )

            # Step 4: LangChain VectorStore — 跳过（LangChainEmbeddingsAdapter 不兼容 langchain-qdrant）
            # 使用原生 Qdrant 客户端做检索（self.qdrant_client）
            self.vector_store = None
            logger.info("✅ Qdrant 原生客户端就绪（跳过 LangChain VectorStore）")

            # ════════════════════════════════════════
            # Step 5: 文档处理 Pipeline
            # ════════════════════════════════════════
            self.pipeline = DocumentProcessingPipeline()
            logger.info("📦 文档处理 Pipeline 初始化完成")

            self._initialized = True
            logger.info("🎉 RAG 服务初始化全部成功！")

        except Exception as e:
            logger.error(f"❌ RAG 初始化失败: {e}", exc_info=True)
            raise

    async def shutdown(self):
        if self.qdrant_client:
            self.qdrant_client.close()
        logger.info("RAG 服务已关闭")

    # ================================================================
    # 文档处理 — 完整 Pipeline 版本
    # ================================================================

    async def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        批量导入目录中的法律文档（使用完整清洗 Pipeline）
        
        流程：文本提取 → 清洗 → 法条级结构解析 → 分块 → 元数据注入 → 向量化入库
        
        支持格式：TXT / MD / PDF / DOCX
        """
        start_time = datetime.now()

        try:
            # 使用 Pipeline 批量处理
            results = await self.pipeline.process_directory(directory_path)

            # 将解析出的 LegalChunk 转换为 LangChain Document 并向量化入库
            all_chunks: List[LegalChunk] = results["all_chunks"]
            ingested_count = 0

            if all_chunks:
                lc_documents = []
                for chunk in all_chunks:
                    doc = LangChainDocument(
                        page_content=chunk.content,
                        metadata=self._build_chunk_metadata(chunk),
                    )
                    lc_documents.append(doc)

                # 批量向量化入库
                self.vector_store.add_documents(lc_documents)
                ingested_count = len(lc_documents)

            elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return {
                "status": "completed",
                "total_files": results["total_files"],
                "success": results["success"],
                "failed": results["failed"],
                "skipped": results["skipped"],
                "total_chunks_parsed": results["total_chunks"],
                "vectors_ingested": ingested_count,
                "errors": results["errors"][:10],  # 只返回前10个错误避免过长
                "elapsed_ms": elapsed_ms,
                "files_detail": results["files_detail"],
            }

        except Exception as e:
            elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"❌ 目录导入失败: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "elapsed_ms": elapsed_ms,
            }

    async def ingest_text(self, text: str, metadata: Dict[str, Any]) -> bool:
        """
        单条文本入库 — 使用 Pipeline 解析分块后再存入
        
        与旧版区别：不再盲目按字符切分，而是先做结构化法条解析
        """
        try:
            # Step 1: 文本清洗
            cleaned_doc = LegalTextCleaner.clean(text)

            # Step 2: 结构解析 & 分块
            filename = metadata.get("source_file", "manual_input")
            chunks = LegalTextParser.parse(cleaned_doc.cleaned_text, filename, metadata)

            # Step 3: 转为 LangChain Document 并入库
            if chunks:
                lc_docs = [
                    LangChainDocument(
                        page_content=c.content,
                        metadata=self._build_chunk_metadata(c),
                    )
                    for c in chunks
                ]
                self.vector_store.add_documents(lc_docs)
                logger.info(f"📥 单条文本入库: {len(chunks)} 个法条分块")
            
            return True

        except Exception as e:
            logger.error(f"单条文本入库失败: {e}", exc_info=True)
            raise

    async def ingest_raw_file(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理单个原始文件（完整 Pipeline）
        
        这是 ingest_directory 的单文件版本，用于上传接口等场景。
        """
        result = await self.pipeline.process_file(file_path, metadata)

        if result["status"] == "ok" and result.get("chunks"):
            lc_docs = [
                LangChainDocument(
                    page_content=chunk.content,
                    metadata=self._build_chunk_metadata(chunk),
                )
                for chunk in result["chunks"]
            ]
            self.vector_store.add_documents(lc_docs)
            result["vectors_ingested"] = len(lc_docs)
        
        return result

    # ================================================================
    # 元数据构建
    # ================================================================

    @staticmethod
    def _build_chunk_metadata(chunk: LegalChunk) -> Dict[str, Any]:
        """将 LegalChunk 转为 Qdrant payload 所需的元数据字典"""
        meta = {
            # === 基础元数据 ===
            "doc_type": chunk.doc_type,
            "file_name": os.path.basename(chunk.source_file) if chunk.source_file else "",
            "source_file": chunk.source_file,
            "ingested_at": datetime.now().isoformat(),

            # === 法律结构元数据（核心！之前缺失的部分）===
            "article_number": chunk.article_number or "",          # 如 "第一条"
            "article_index": chunk.article_index,                  # 数字序号
            "chapter": chunk.chapter or "",                       # 所属章
            "section": chunk.section or "",                       # 所属节
            "law_name": chunk.law_name or "",                     # 法律名称

            # === 统计信息 ===
            "char_length": len(chunk.content),

            # === 额外信息 ===
        }

        # 合并 extra_metadata
        if chunk.extra_metadata:
            meta.update({k: v for k, v in chunk.extra_metadata.items() if k not in meta})

        return meta

    # ================================================================
    # 检索
    # ================================================================

    async def search(
        self,
        query: str,
        top_k: int = 5,
        scope: Optional[str] = None,
        score_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """混合语义检索 — 使用 Qdrant 原生客户端"""
        import requests as _requests

        # 1. 查询向量化
        resp = _requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {settings.EMBEDDING_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": settings.EMBEDDING_MODEL, "input": [query]},
            timeout=30,
        )
        resp.raise_for_status()
        query_vec = resp.json()["data"][0]["embedding"]

        # 2. Qdrant 搜索
        query_filter = None
        if scope and scope != "all":
            query_filter = Filter(
                must=[FieldCondition(key="doc_type", match=MatchValue(value=scope))]
            )

        results = self.qdrant_client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vec,
            limit=top_k,
            query_filter=query_filter,
        )

        # 3. 构造返回
        search_results = []
        for r in results:
            score = r.score  # 余弦相似度（已归一化向量，点积即相似度）
            if score >= score_threshold:
                payload = r.payload
                search_results.append({
                    "content": payload.get("content", ""),
                    "metadata": {
                        "doc_type": payload.get("doc_type", ""),
                        "file_name": payload.get("file_name", ""),
                        "law_name": payload.get("law_name", ""),
                        "article_number": payload.get("article_number", ""),
                        "chapter": payload.get("chapter", ""),
                    },
                    "relevance_score": float(score),
                })

        return search_results

    # ================================================================
    # LLM 生成
    # ================================================================

    async def generate(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        task_type: str = "legal_research",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """基于检索上下文 + LLM 生成回答"""
        # 组装上下文（带丰富引用信息）
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            meta = chunk.get("metadata", {})
            law_name = meta.get("law_name", meta.get("file_name", "未知"))
            article_num = meta.get("article_number", "")
            chapter = meta.get("chapter", "")

            source_label = law_name
            if article_num:
                source_label += f" · {article_num}"
            if chapter:
                source_label += f" ({chapter})"

            context_text += f"\n--- 【引用{i}】{source_label} ---\n"
            context_text += chunk["content"]
            context_text += f"\n[相关度: {chunk['relevance_score']:.2f}]\n"

        user_prompt = f"""## 用户问题
{question}

## 相关法律法规/案例（已通过语义检索找到以下最相关的条文）
{context_text if context_text else "未检索到高度相关的法律资料，请基于你的法律知识谨慎回答。"}

## 请基于上述参考资料，按照系统要求的格式进行专业分析。注意：
- 必须准确引用上述参考资料的条文编号和内容
- 如果参考资料不足以回答，明确说明需要补充什么信息
"""

        messages = [
            ("system", LAW_SYSTEM_PROMPT),
            ("human", user_prompt),
        ]

        if chat_history:
            for turn in chat_history[-12:]:
                messages.append(tuple(turn.items()))

        response = await self.llm.ainvoke(messages)
        return response.content

    # ================================================================
    # 工具方法
    # ================================================================

    def _detect_doc_type(self, filename: str) -> str:
        """根据文件名检测文档类型"""
        name_lower = filename.lower()
        law_keywords = ["法", "条例", "规定", "办法", "解释", "决定", "规则"]
        case_keywords = ["案号", "判决书", "裁定书", "判决", "(20"]

        for kw in law_keywords:
            if kw in filename:
                return "law"
        for kw in case_keywords:
            if kw in filename:
                return "case"
        if "合同" in filename or "协议" in filename:
            return "agreement"
        return "other"

    @property
    def is_ready(self) -> bool:
        return self._initialized

    async def get_collection_info(self) -> Dict[str, Any]:
        if not self.qdrant_client:
            return {"status": "not_initialized"}
        try:
            info = self.qdrant_client.get_collection(settings.QDRANT_COLLECTION)
            
            # 获取 Embedding 服务信息
            embedding_info = {}
            if self.embedding_service:
                embedding_info = self.embedding_service.get_info()
            
            return {
                "status": "ready",
                "collection": settings.QDRANT_COLLECTION,
                "vectors_count": info.points_count,
                "segments_count": info.segments_count,
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance.value),
                
                # ⭐ 新增：Embedding 模型信息
                "embedding": embedding_info,
                
                "pipeline_enabled": True,
                "features": [
                    "多格式文本提取(TXT/MD/PDF/DOCX)",
                    "法律文本专用清洗(断行修复/去噪/标点规范化)",
                    "法条级智能分块(第X条模式识别)",
                    "章节层级结构解析",
                    "丰富元数据注入(条号/章/节/法律名)",
                    "专用 Embedding 模型(非 LLM 通用模型)",
                ],
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
