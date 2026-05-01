"""
AtomicKnowledgeExtractor v2 — 规则版（无需 LLM）

用正则 + 关键词表从法条原文中提取结构化字段。
速度快（毫秒级），零成本，覆盖 80%+ 常见法条结构。

提取字段: subject / behavior / condition / legal_effect / exception / object_ / keywords
"""

import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


# ===== 主体词典 =====
SUBJECTS = [
    "公司", "股东", "董事", "监事", "高级管理人员",
    "债权人", "债务人", "保证人", "抵押人", "出质人",
    "劳动者", "用人单位", "雇主", "雇员",
    "人民法院", "人民检察院", "仲裁机构",
    "申请人", "被申请人", "原告", "被告", "第三人",
    "消费者", "经营者", "生产者", "销售者",
    "合伙人", "破产管理人", "清算组",
    "发包人", "承包人", "出租人", "承租人",
    "委托人", "受托人", "代理人",
    "保险公司", "投保人", "被保险人", "受益人",
    "证券公司", "上市公司", "发行人",
    "行政机关", "行政机关负责人",
    "当事人", "利害关系人", "相关人员",
]

# ===== 行为关键词 =====
BEHAVIOR_PREFIXES = [
    "不得", "可以", "应当", "必须",
    "有权", "无权", "可", "应",
]


def extract_article(text: str, law_name: str = "", article_number: str = "") -> Dict[str, Any]:
    """
    从法条原文中提取结构化字段

    返回:
      {subject, behavior, condition, legal_effect, exception, object_, keywords}
    """
    if not text or len(text) < 5:
        return {}

    text_clean = text.strip()

    # 去掉开头的"第X条"（如果存在）
    text_clean = re.sub(r"^第[^条]+条\s*", "", text_clean).strip()

    result = {}

    # 1. 提取但书/例外（先提取，从原文中移除，避免干扰其他字段）
    exception = _extract_exception(text_clean)
    result["exception"] = exception

    main_text = text_clean
    if exception:
        main_text = text_clean.replace(exception, "。", 1)

    # 2. 提取条件
    condition = _extract_condition(main_text)
    result["condition"] = condition

    # 3. 提取主体
    subject = _extract_subject(main_text)
    result["subject"] = subject

    # 4. 提取行为 + 法律效果
    behavior_text = _extract_behavior_text(main_text, condition)
    behavior, legal_effect = _parse_behavior_and_effect(behavior_text)
    result["behavior"] = behavior
    result["legal_effect"] = legal_effect

    # 5. 提取客体
    object_ = _extract_object(behavior_text, behavior)
    result["object_"] = object_

    # 6. 提取关键词
    result["keywords"] = _extract_keywords(text_clean, subject, behavior)

    return result


# ===== 但书/例外提取 =====

EXCEPTION_PATTERNS = [
    r"(?:但是|但)[^。]*(?:除外|不在此限|不适用本|另有规定|例外)",
    r"[^。]*(?:另有规定的除外|另有约定的除外|另有规定的，从其规定)",
    r"[^。]*(?:不适用前款规定|不适用本条)",
    r"[^。]*但是[^。]*",
]


def _extract_exception(text: str) -> str:
    """提取但书/例外条款"""
    for pattern in EXCEPTION_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group().strip()
    return ""


# ===== 条件提取 =====

CONDITION_PATTERNS = [
    # 经XXX
    r"(?:经|依照|按照|根据)[^，。]{2,50}(?:的|后)，",
    # 有下列情形
    r"有(?:下列|以下)[^，。]{2,30}(?:情形之一的?|情形)[^，。]{0,30}",
    # 有前款
    r"有前款[^，。]{2,30}",
    # 在XXX情况下
    r"(?:在|于)[^，。]{2,50}(?:时|前|内|后|中|的)，",
    # 因XXX
    r"因[^，。]{2,80}(?:的，|，)",
    # 如果XXX
    r"如果[^，。]{2,80}，",
    # 未XXX
    r"未[^，。]{2,60}(?:的，|，)",
    # 超过XXX
    r"超过[^，。]{2,40}(?:的，|，)",
    # 当XXX
    r"当[^，。]{2,40}(?:时|后)，",
]


def _extract_condition(text: str) -> str:
    """提取条件句"""
    # 找到第一个条件句（句子结束是，或。）
    conditions = []
    for pattern in CONDITION_PATTERNS:
        m = re.search(pattern, text)
        if m:
            c = m.group().strip()
            # 去重
            if c not in conditions:
                conditions.append(c)
    return "；".join(conditions[:3]) if conditions else ""


# ===== 主体提取 =====

def _extract_subject(text: str) -> str:
    """提取法律主体"""
    # 按长度排序（长匹配优先）
    subjects_sorted = sorted(SUBJECTS, key=len, reverse=True)
    for subj in subjects_sorted:
        if subj in text:
            return subj
    return ""


# ===== 行为 + 法律效果提取 =====

BEHAVIOR_END_MARKERS = ["。", "；", "但是", "但", "，但"]


def _extract_behavior_text(text: str, condition: str) -> str:
    """
    提取包含行为描述的核心句（去掉条件句之后的部分）
    """
    core = text
    if condition:
        core = core.replace(condition, "", 1).strip()
        core = re.sub(r"^，", "", core).strip()

    # 取第一句
    first_sentence = core.split("。")[0] if "。" in core else core
    return first_sentence.strip()


def _parse_behavior_and_effect(text: str) -> tuple:
    """
    从行为句中解析行为和法律效果

    "公司不得收购本公司股份" → ("收购本公司股份", "不得")
    "公司可以发行股份" → ("发行股份", "可以")
    "公司应当承担赔偿责任" → ("承担赔偿责任", "应当")
    """
    for prefix in BEHAVIOR_PREFIXES:
        idx = text.find(prefix)
        if idx >= 0:
            behavior = text[idx + len(prefix):].strip()
            legal_effect = prefix
            # 清理多余标点
            behavior = behavior.strip("，。；")
            if behavior:
                return behavior, legal_effect

    # 没有前缀词，取第一个动词短语
    return text[:50], ""


# ===== 客体提取 =====

OBJECT_MARKERS = [
    "本公司股份", "股权", "债权", "债务", "财产", "抵押物",
    "质押物", "保证金", "出资额", "股份", "股票", "债券",
    "商标", "专利", "著作权", "商业秘密",
    "土地使用权", "房产", "机动车",
    "合同", "协议", "订单",
    "基金", "信托", "保险单",
    "货物", "产品", "商品",
]

OBJECT_PATTERNS = [
    r"(?:收购|转让|抵押|质押|处置|拍卖|变卖)([^，。；]{2,20})",
    r"[^，。；]{0,10}(?:的)([^，。；]{2,15})(?:的)",
]


def _extract_object(text: str, behavior: str) -> str:
    """提取法律客体"""
    if not behavior:
        return ""

    # 从行为句中提取客体
    for marker in OBJECT_MARKERS:
        if marker in behavior:
            return marker

    # 正则匹配
    for pattern in OBJECT_PATTERNS:
        m = re.search(pattern, behavior)
        if m:
            return m.group(1).strip()
    return ""


# ===== 关键词提取 =====

KEYWORD_DICT = [
    # 公司治理
    "公司", "股东", "股权", "董事", "监事", "高管",
    "股东大会", "董事会", "监事会", "表决权", "决议",
    "出资", "增资", "减资", "回购", "转让", "合并", "分立",
    "解散", "清算", "破产", "重整",
    # 合同
    "合同", "契约", "协议", "要约", "承诺",
    "违约", "解除", "撤销", "无效", "效力",
    "赔偿", "违约金", "定金", "担保",
    # 债权债务
    "债权", "债务", "债权人", "债务人",
    "抵押", "质押", "留置", "保证",
    # 诉讼
    "诉讼", "仲裁", "管辖", "起诉", "上诉", "再审",
    "执行", "保全", "证据", "举证", "质证",
    # 劳动
    "劳动", "劳动合同", "工资", "社保",
    "辞退", "裁员", "补偿", "赔偿金",
    # 知识产权
    "商标", "专利", "著作权", "侵权",
    # 刑事
    "犯罪", "刑罚", "有期徒刑", "罚金",
    "自首", "立功", "缓刑", "减刑",
]

STOP_WORDS = {"的", "了", "在", "是", "有", "和", "与", "或",
               "对", "为", "由", "于", "以", "及", "被", "把",
               "从", "到", "让", "向", "往", "用", "按", "照"}


def _extract_keywords(text: str, subject: str, behavior: str) -> List[str]:
    """提取关键词"""
    keywords = set()

    # 主体作为关键词
    if subject and len(subject) > 1:
        keywords.add(subject)

    # 行为中的关键词
    if behavior:
        for word in KEYWORD_DICT:
            if word in behavior:
                keywords.add(word)

    # 全文关键词匹配
    for word in KEYWORD_DICT:
        if word in text:
            keywords.add(word)
        if len(keywords) >= 6:
            break

    # 去掉停用词和单字
    result = [k for k in keywords if k not in STOP_WORDS and len(k) > 1]

    return result[:6]


# ===== 工具 =====

def article_index(article_number: str) -> int:
    """中文条文编号 → 数字"""
    if not article_number:
        return 0
    digits = re.findall(r"\d+", article_number)
    if digits:
        return int(digits[0])
    # 中文数字
    cn_nums = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
               "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    cn_levels = {"十": 10, "百": 100, "千": 1000}
    result, temp = 0, 0
    for char in article_number:
        if char in cn_nums:
            temp += cn_nums[char]
        elif char in cn_levels:
            if temp == 0:
                temp = cn_levels[char]
            else:
                temp *= cn_levels[char]
        else:
            result += temp
            temp = 0
    result += temp
    return result


def to_qdrant_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """转为 Qdrant payload 格式"""
    payload = {}
    for field in ["subject", "behavior", "condition", "legal_effect",
                   "exception", "object_"]:
        v = result.get(field, "")
        if v:
            payload[field] = v
    kw = result.get("keywords", [])
    if kw:
        payload["keywords"] = kw
    payload["extracted_by"] = "rule_based_v2"
    return payload
