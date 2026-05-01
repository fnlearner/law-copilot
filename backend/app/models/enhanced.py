"""
LawCopilot 增强版 RAG — 数据模型 v2.0

增强内容：
  - AtomicLegalKnowledge: 法规结构化原子知识
  - JudgmentTriple: 裁判文书争议焦点三元组
  - VerifiedCitation: 经过校验的可信引用
  - 多集合 Collection 配置
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


# ===== 枚举 =====

class SourceType(str, Enum):
    """知识来源类型"""
    LAW = "law"                     # 法律法规
    JUDGMENT = "judgment"           # 裁判文书
    COMMENTARY = "commentary"       # 权威评注
    REGULATION = "regulation"       # 行政法规
    JUDICIAL_INTERP = "judicial_interp"  # 司法解释
    LOCAL_LAW = "local_law"         # 地方法规
    GUIDE_CASE = "guide_case"       # 指导案例


class AuthorityLevel(str, Enum):
    """权威性等级"""
    CONSTITUTION = 1.0               # 宪法
    LAW = 0.95                       # 法律
    JUDICIAL_INTERP = 0.90           # 司法解释
    ADMIN_REGULATION = 0.85          # 行政法规
    GUIDE_CASE = 0.90                # 指导案例（权威性高）
    BULLETIN_CASE = 0.80             # 公报案例
    LOCAL_REGULATION = 0.60          # 地方法规
    ORDINARY_CASE = 0.50             # 普通案例
    COMMENTARY = 0.40                # 学者评注


class CitationStatus(str, Enum):
    """引用校验状态"""
    VERIFIED = "verified"            # 在知识库中精确定位到
    FUZZY_MATCH = "fuzzy_match"      # 模糊匹配到相似条文
    UNVERIFIED = "unverified"        # 知识库中不存在需要人工确认


# ===== Qdrant 集合配置 =====

@dataclass
class CollectionConfig:
    """Qdrant 集合配置"""
    name: str
    description: str
    vector_size: int = 512              # BGE-small-zh 维度
    distance: str = "Cosine"
    chunk_strategy: str = ""
    source: str = ""

    def to_qdrant_config(self) -> Dict[str, Any]:
        from qdrant_client.models import VectorParams, Distance
        return {
            "vectors_config": VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        }


# 定义所有集合
COLLECTION_CONFIGS: Dict[str, CollectionConfig] = {
    "laws": CollectionConfig(
        name="laws",
        description="法律法规条文（含结构化原子知识）",
        chunk_strategy="按第X条切分 + 结构化提取",
        source="LawRefBook + FLK",
    ),
    "judgments": CollectionConfig(
        name="judgments",
        description="裁判文书（争议焦点三元组）",
        chunk_strategy="争议焦点-法院观点-法条依据 三元组",
        source="裁判文书网 / 最高法公报",
    ),
    "commentaries": CollectionConfig(
        name="commentaries",
        description="权威评注（法条释义/理解与适用）",
        chunk_strategy="按法条编号映射",
        source="人大版法律释义 / 最高法理解与适用",
    ),
}


# ===== 结构化原子知识（法规类） =====

@dataclass
class AtomicLegalKnowledge:
    """
    法规的结构化原子知识单元

    传统 RAG 检索到的是"法条文本块"，
    增强版检索到的是结构化的法律知识单元。

    示例:
      法律: 《中华人民共和国公司法》第一百四十二条
      subject: "公司"
      behavior: "回购股份"
      condition: "为减少公司注册资本，经股东大会决议"
      legal_effect: "可以收购本公司股份"
      exception: "但属于第(一)(二)项情形的，应当经股东大会决议"
    """

    # === 定位信息 ===
    law_name: str                                    # 《中华人民共和国公司法》
    article_number: str                              # "第一百四十二条"
    article_index: int                               # 142（数字序号，方便排序）
    chapter: str = ""                                # "第五章 股份有限公司的股份发行和转让"
    section: str = ""                                # "第一节 股份发行"
    version_date: str = ""                           # "2023-12-29"
    status: str = "有效"                              # "有效" / "已修改" / "已废止"

    # === 原子知识（核心！）===
    subject: str = ""                                # 主体: "公司" / "股东" / "董事"
    behavior: str = ""                               # 行为: "回购股份" / "表决" / "出资"
    condition: str = ""                              # 条件: "经股东大会决议"
    legal_effect: str = ""                           # 法律效果: "可以...应当..."
    exception: str = ""                              # 但书/例外: "但是..."
    object_: str = ""                                # 客体: "股份" / "股权" / "债权"

    # === 关联信息 ===
    related_articles: List[str] = field(default_factory=list)   # 关联条文
    amendment_history: List[str] = field(default_factory=list)  # 修改历史
    keywords: List[str] = field(default_factory=list)           # 关键词标签

    # === 元数据 ===
    source_type: SourceType = SourceType.LAW
    authority_level: AuthorityLevel = AuthorityLevel.LAW
    source_file: str = ""
    ingested_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # === 原文 ===
    full_text: str = ""                              # 法条完整原文

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_embedding_text(self) -> str:
        """用于向量化的文本（拼接关键字段）"""
        parts = [
            f"【{self.law_name}】",
            f"第{self.article_number}条" if self.article_number else "",
            f"主体：{self.subject}" if self.subject else "",
            f"行为：{self.behavior}" if self.behavior else "",
            f"条件：{self.condition}" if self.condition else "",
            f"效果：{self.legal_effect}" if self.legal_effect else "",
            f"例外：{self.exception}" if self.exception else "",
            self.full_text[:500],
        ]
        return "\n".join(p for p in parts if p)

    @property
    def article_id(self) -> str:
        """唯一标识：law_name + article_number"""
        return f"{self.law_name}::{self.article_number}"


# ===== 裁判文书三元组 =====

@dataclass
class JudgmentTriple:
    """
    裁判文书争议焦点三元组

    传统做法：整篇裁判文书 embedding（噪声大、精度低）
    增强做法：提取 (争议焦点, 法院观点, 法条依据) 三元组

    检索时，用户可通过任意一个入口命中：
      - 争议焦点 → 返回说理 + 法条
      - 法院观点 → 返回焦点 + 法条
      - 法条依据 → 返回焦点 + 说理
    """

    # === 案件信息 ===
    case_number: str                                 # "(2023)最高法民终XX号"
    case_name: str                                   # "XX公司与XX公司股权转让纠纷案"
    court: str                                       # "最高人民法院"
    judgment_date: str = ""                          # "2023-05-20"
    case_type: str = ""                              # "民事" / "行政" / "刑事"
    trial_level: str = ""                            # "一审" / "二审" / "再审"

    # === 核心三元组 ===
    dispute_focus: str = ""                          # 争议焦点
    court_opinion: str = ""                          # 法院观点（说理部分）
    legal_basis: List[str] = field(default_factory=list)   # 法条依据列表

    # === 裁判结果 ===
    verdict: str = ""                                # "驳回上诉，维持原判"
    key_ratio: str = ""                              # 裁判要旨（摘要）

    # === 元数据 ===
    source_type: SourceType = SourceType.JUDGMENT
    authority_level: AuthorityLevel = AuthorityLevel.GUIDE_CASE
    keywords: List[str] = field(default_factory=list)
    ingested_at: str = field(default_factory=lambda: datetime.now().isoformat())
    full_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_embedding_text(self) -> str:
        """用于向量化的文本"""
        parts = [
            f"【{self.case_name}】({self.case_number})",
            f"争议焦点：{self.dispute_focus}",
            f"法院观点：{self.court_opinion[:300]}",
            f"法条依据：{'；'.join(self.legal_basis)}",
            f"裁判要旨：{self.key_ratio}",
        ]
        return "\n".join(p for p in parts if p)


# ===== 校验引用 =====

@dataclass
class VerifiedCitation:
    """
    经过校验的可信引用

    在 LLM 生成回答后，用此结构对每条引用进行反向验证。
    确保不会出现 LLM 虚构法条的情况。
    """

    # === 引用原文 ===
    raw_text: str                                    # LLM 输出的引用文本
    law_name: str                                    # 提取的法律名称
    article_number: str                              # 提取的条文编号

    # === 校验结果 ===
    status: CitationStatus                           # 校验状态
    confidence: float = 0.0                          # 置信度
    matched_content: str = ""                        # 知识库中匹配到的原文
    qdrant_point_id: str = ""                        # Qdrant 中的 point ID
    qdrant_collection: str = ""                      # 所在的集合

    # === 格式化 ===
    def format_display(self, verbose: bool = False) -> str:
        """格式化显示引用"""
        icons = {
            CitationStatus.VERIFIED: "✅",
            CitationStatus.FUZZY_MATCH: "🔶",
            CitationStatus.UNVERIFIED: "⚠️",
        }
        icon = icons.get(self.status, "❓")
        text = f"{icon} {self.raw_text}"
        if verbose and self.matched_content:
            text += f"\n   → 原文: {self.matched_content[:100]}..."
        return text


# ===== 检索结果 =====

@dataclass
class ScoredChunk:
    """
    检索结果（含多路融合分）

    经过三路检索 + 重排序后，每个结果带有:
      - 语义分（向量检索）
      - BM25 分（关键词）
      - 结构分（字段精确匹配）
      - 权威分（doc_type 权重）
      - 时效分（公布日期近的加分）
      - 融合分（最终排序依据）
    """
    # 内容
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 检索分数
    vector_score: float = 0.0
    bm25_score: float = 0.0
    struct_score: float = 0.0

    # 综合权重
    authority_score: float = 0.0
    timeliness_score: float = 0.0
    rerank_score: float = 0.0          # Cross-encoder 重排分
    final_score: float = 0.0           # 最终融合分

    # 来源
    collection: str = ""                # 来自哪个集合
    source_type: SourceType = SourceType.LAW
    point_id: str = ""
