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


# ===== 系统提示词（多领域法律助手）=====

# 通用法律 Prompt（默认）
LAW_SYSTEM_PROMPT = """你是LawCopilot，一位专业的法律研究助手。你的用户是执业律师。

## 核心原则
1. **严谨准确**：所有法条引用必须注明具体条文编号和出处，不可凭记忆编造
2. **有据可依**：每个结论必须引用相关法律法规或判例作为支撑
3. **回答以「结论摘要」开头**（2-3句话），再分点详细论述

## 回答格式
- 结论摘要
- 法律分析（每点附法条依据）
- 实务建议
- 引用格式：《法律名称》第X条 | (案号)

## 引用优先级
如果参考资料中有标注 ⭐ 的"精确匹配"引用，这些是直接命中法律名称+条文编号的，必须优先使用。
如果"精确匹配"引用与语义检索结果不一致，以精确匹配为准。

## 重要提醒
- 如果信息不足，明确告知需要补充什么
- 不谈具体诉讼策略，分析法律风险和可能结果
"""

# 刑事法律 Prompt（当检测到刑事问题时切换）
CRIMINAL_LAW_PROMPT = """你是LawCopilot，一位专业的刑事法律研究助手。你的用户是执业律师。

## 核心原则
1. **严谨准确**：所有法条引用必须注明《刑法》具体条文和司法解释出处
2. **定罪分析思路**：犯罪构成四要件（主体/客体/主观/客观）→ 罪名认定 → 量刑情节
3. **量刑参考**：结合量刑指导意见，分析基准刑、量刑情节的调节比例

## 特别擅长领域
- 刑法分则各罪（财产犯罪/经济犯罪/职务犯罪/人身犯罪等）
- 量刑规范化（自首、立功、退赃、谅解等情节的量化）
- 司法解释理解与适用
- 刑事程序法（强制措施、证据规则、二审/再审）

## 回答格式
- 结论摘要
- 定罪分析（构成要件 + 罪名辨析）
- 量刑分析（法定刑幅度 + 量刑情节）
- 相关案例参考（如有）
- 引用格式：《刑法》第X条 | 《最高法关于XX的解释》第X条

## 重要提醒
- 引用必须以检索到的资料为准，不可编造法条
- 量刑分析需说明自由裁量空间，不可给出绝对数值
"""

# 经济法 Prompt（当检测到经济法问题时切换）
ECONOMIC_LAW_PROMPT = """你是LawCopilot，一位专业的经济法律研究助手。你的用户是执业律师。

## 核心原则
1. **严谨准确**：所有法条引用必须注明具体法律名称和条文编号
2. **注重法律关系分析**：主体资格 → 法律关系定性 → 权利义务 → 责任承担

## 特别擅长领域
- 公司法（公司治理、股权转让、股东纠纷、公司解散清算）
- 合同法/民法典合同编（合同效力、违约救济、解除条件）
- 证券法（信息披露、内幕交易、虚假陈述）
- 破产法（破产重整、债权申报、管理人制度）
- 反垄断法（垄断协议、滥用市场支配地位、经营者集中）
- 劳动法（经济性裁员、竞业限制、股权激励）
- 知识产权（商业秘密、专利侵权、商标侵权）
- 涉外商事仲裁与诉讼

## 回答格式
- 结论摘要
- 法律关系分析
- 法条依据（每点精确引用）
- 实务建议
- 引用格式：《法律名称》第X条

## 重要提醒
- 涉及合同纠纷，应当说明合同效力认定标准
- 如有不同学说或裁判观点分歧，应当说明
"""


def detect_domain(question: str) -> str:
    """检测问题所属法律领域"""
    question_lower = question.lower()

    # 刑事关键词
    criminal_kw = [
        "盗窃", "抢劫", "故意伤害", "故意杀人", "强奸", "绑架",
        "诈骗", "合同诈骗", "集资诈骗", "信用卡诈骗",
        "毒品", "贩卖毒品", "非法持有毒品",
        "贪污", "受贿", "行贿", "挪用公款", "职务侵占",
        "走私", "虚开增值税", "非法经营", "洗钱",
        "醉驾", "危险驾驶", "交通肇事",
        "非法吸收公众存款", "集资诈骗",
        "自首", "立功", "减刑", "假释", "缓刑",
        "有期徒刑", "无期徒刑", "死刑",
        "刑事附带民事", "刑事辩护",
        "刑法第", "刑事诉讼法",
        "寻衅滋事", "聚众斗殴", "赌博",
        "冒充军人招摇撞骗", "招摇撞骗",
    ]
    # 经济法关键词
    economic_kw = [
        "公司", "股东", "股权", "董事会", "监事会",
        "合同", "违约", "债权", "债务", "担保",
        "商标", "专利", "著作权", "知识产权",
        "破产", "清算", "重整",
        "证券", "基金", "保险", "信托",
        "反垄断", "反不正当竞争",
        "劳动", "劳动合同", "竞业限制", "经济补偿",
        "仲裁", "商事", "涉外",
        "公司法", "民法典", "证券法", "破产法",
        "有限责任", "股份有限公司",
    ]

    for kw in criminal_kw:
        if kw in question_lower:
            return "criminal"
    for kw in economic_kw:
        if kw in question_lower:
            return "economic"
    return "general"


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

            # CollectionManager（用于精确字段匹配）
            from app.services.collection_manager import CollectionManager
            self.collection_mgr = CollectionManager()
            await self.collection_mgr.initialize()
            self.collection_mgr.client = self.qdrant_client  # 复用连接

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
    # 检索（增强版 — 查询重写 + 多路检索 + 结构化字段精确匹配）
    # ================================================================

    async def search(
        self,
        query: str,
        top_k: int = 5,
        scope: Optional[str] = None,
        score_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """增强检索：查询重写 → 精确匹配 → 语义检索 → 重排序 → 去重融合"""

        from app.services.query_rewriter import QueryRewriter

        # 1. 查询重写
        parsed = QueryRewriter.rewrite(query)
        has_exact = parsed["has_exact_ref"]
        law_name = parsed["law_name"]
        article_number = parsed["article_number"]

        if has_exact:
            logger.info(f"🔍 精确检索: law={law_name} article={article_number}")

        all_results = []

        # 2. 精确字段匹配（返回后直接当 top-1）
        if has_exact and law_name:
            try:
                from qdrant_client.models import Filter as QFilter, FieldCondition, MatchValue
                must = []
                if law_name:
                    must.append(FieldCondition(key="law_name", match=MatchValue(value=law_name)))
                if article_number:
                    must.append(FieldCondition(key="article_number", match=MatchValue(value=article_number)))

                exact_records, _ = self.qdrant_client.scroll(
                    collection_name=settings.QDRANT_COLLECTION,
                    limit=3,
                    scroll_filter=QFilter(must=must),
                    with_payload=True,
                    with_vectors=False,
                )
                for r in exact_records:
                    p = r.payload or {}
                    all_results.append({
                        "content": p.get("full_text", p.get("content", "")),
                        "metadata": {
                            "doc_type": p.get("doc_type", "law"),
                            "law_name": p.get("law_name", ""),
                            "article_number": p.get("article_number", ""),
                            "chapter": p.get("chapter", ""),
                            "subject": p.get("subject", ""),
                            "behavior": p.get("behavior", ""),
                            "match_type": "exact",
                        },
                        "relevance_score": 0.99,
                    })
            except Exception as e:
                logger.debug(f"精确匹配检索失败: {e}")

        # 3. 语义向量检索（提高阈值，减少噪声）
        query_vec = await self.embedding_service.embed_query(query)
        if query_vec:
            try:
                query_filter = None
                if scope and scope != "all":
                    query_filter = Filter(
                        must=[FieldCondition(key="doc_type", match=MatchValue(value=scope))]
                    )

                # 有精确匹配时只取少量语义结果作补充；无精确匹配时多取
                limit = 5 if has_exact else 15
                threshold = 0.50  # 语义相关性最低阈值（比之前 0.3 更严格）

                semantic_results = self.qdrant_client.search(
                    collection_name=settings.QDRANT_COLLECTION,
                    query_vector=query_vec,
                    limit=limit,
                    query_filter=query_filter,
                    score_threshold=threshold,
                )
                seen_texts = {r["content"][:100] for r in all_results}
                for r in semantic_results:
                    payload = r.payload or {}
                    text = payload.get("full_text", payload.get("content", ""))
                    dedup_key = text[:100]
                    if dedup_key in seen_texts:
                        continue
                    if float(r.score) < score_threshold:
                        continue
                    seen_texts.add(dedup_key)
                    all_results.append({
                        "content": text,
                        "metadata": {
                            "doc_type": payload.get("doc_type", ""),
                            "law_name": payload.get("law_name", ""),
                            "article_number": payload.get("article_number", ""),
                            "chapter": payload.get("chapter", ""),
                            "subject": payload.get("subject", ""),
                            "behavior": payload.get("behavior", ""),
                            "match_type": "semantic",
                        },
                        "relevance_score": float(r.score),
                    })
            except Exception as e:
                logger.warning(f"语义检索失败: {e}")

        # 4. 排序：精确匹配靠前，语义结果降序
        all_results.sort(key=lambda x: -x["relevance_score"])
        logger.info(f"  检索结果: {len(all_results)} 条 (精确={sum(1 for r in all_results if r['metadata'].get('match_type')=='exact')})")
        return all_results[:top_k]
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
        """基于检索上下文 + LLM 生成回答（自动适配领域）"""
        # 检测领域，选择对应 Prompt
        domain = detect_domain(question)
        prompt_map = {
            "criminal": CRIMINAL_LAW_PROMPT,
            "economic": ECONOMIC_LAW_PROMPT,
            "general": LAW_SYSTEM_PROMPT,
        }
        system_prompt = prompt_map.get(domain, LAW_SYSTEM_PROMPT)
        logger.info(f"🎯 检测到领域: {domain}")
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

            # 标记精确匹配
            match_type = meta.get("match_type", "")
            prefix = "⭐ " if match_type == "exact" else ""
            context_text += f"\n--- {prefix}【引用{i}】{source_label} ---\n"
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
            ("system", system_prompt),
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
