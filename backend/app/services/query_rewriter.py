"""
QueryRewriter — 用户查询重写 + 结构化提取

策略:
  1. LLM 查询规划（主流程）: 用 LLM 理解用户意图，生成搜索策略
  2. 规则快速路径（兜底）: 当 LLM 不可用时，用正则+词典做基本提取

LLM 查询规划的输出:
  - domain: 法律领域 (civil/criminal/economic/admin/procedural/general)
  - law_name: 目标法律（如果明确提及）
  - article_number: 目标条文（如果明确提及）
  - sub_queries: 用于语义检索的多条子查询
  - search_strategy: "exact" / "semantic" / "hybrid"
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ===== 法律名称词典（仅用于规则快速路径）=====
LAW_NAMES = {
    "公司法": "中华人民共和国公司法",
    "民法典": "中华人民共和国民法典",
    "刑法": "中华人民共和国刑法",
    "民事诉讼法": "中华人民共和国民事诉讼法",
    "刑事诉讼法": "中华人民共和国刑事诉讼法",
    "劳动合同法": "中华人民共和国劳动合同法",
    "证券法": "中华人民共和国证券法",
    "反垄断法": "中华人民共和国反垄断法",
    "破产法": "中华人民共和国企业破产法",
    "著作权法": "中华人民共和国著作权法",
    "商标法": "中华人民共和国商标法",
    "专利法": "中华人民共和国专利法",
    "担保法": "中华人民共和国担保法",
    "劳动法": "中华人民共和国劳动法",
    "消费者权益保护法": "中华人民共和国消费者权益保护法",
    "环境保护法": "中华人民共和国环境保护法",
    "噪声污染防治法": "中华人民共和国噪声污染防治法",
    "道路交通安全法": "中华人民共和国道路交通安全法",
    "行政处罚法": "中华人民共和国行政处罚法",
    "行政复议法": "中华人民共和国行政复议法",
    "合同法": "中华人民共和国民法典",
    "婚姻法": "中华人民共和国民法典",
    "继承法": "中华人民共和国民法典",
}

# LLM 查询规划的系统提示
QUERY_PLAN_PROMPT = """你是一名法律检索专家。用户输入的是一个法律咨询问题。
请分析用户的查询，输出一个 JSON 格式的检索计划，用于在法律法规数据库中搜索相关资料。

请严格按以下 JSON 格式输出，不要包含其他内容：
{
  "domain": "法律领域 (civil/criminal/economic/admin/procedural/general)",
  "intent": "用户的核心法律诉求（一句话概括）",
  "law_name": "明确提及的法律名称（如未提及则为空字符串）",
  "article_number": "明确提及的条文编号（如未提及则为空字符串）",
  "sub_queries": ["用于语义检索的子查询，3-5条，覆盖不同角度"],
  "search_strategy": "exact(精确匹配) / semantic(语义检索) / hybrid(混合)",
  "reasoning": "为什么这样规划检索策略"
}
"""


class QueryRewriter:
    """查询重写器（LLM 规划 + 规则兜底）"""

    # 缓存 LLM 结果避免重复调用
    _llm_cache: Dict[str, Dict] = {}

    @staticmethod
    def rewrite(query: str, llm_func=None) -> Dict[str, Any]:
        """
        重写用户查询，返回结构化检索参数。

        Args:
            query: 用户原始查询
            llm_func: 可选的 LLM 调用函数，用于查询规划。
                      签名: llm_func(system_prompt, user_query) -> str
                      如果不提供，则走规则快速路径。

        Returns:
            {
                "original": 原始查询,
                "domain": 法律领域,
                "intent": 用户核心诉求,
                "law_name": 目标法律全称或 "",
                "article_number": 目标条文或 "",
                "article_index": 数字条文号或 0,
                "legal_terms": [核心法律概念],
                "sub_queries": [用于检索的子查询],
                "has_exact_ref": 是否包含精确法条引用,
                "search_strategy": "exact" / "semantic" / "hybrid",
                "plan_source": "llm" / "rule",
                "reasoning": "规划依据",
            }
        """
        # 先快速判断是否可以直接走规则（精确法条引用）
        quick = QueryRewriter._quick_parse(query)
        if quick["has_exact_ref"]:
            quick["plan_source"] = "rule"
            quick["intent"] = ""
            quick["reasoning"] = "用户提供了精确法条引用，走精确匹配"
            return quick

        # 尝试 LLM 规划
        if llm_func is not None:
            try:
                plan = QueryRewriter._llm_plan(query, llm_func)
                if plan:
                    plan["original"] = query
                    plan["has_exact_ref"] = bool(plan.get("law_name") or plan.get("article_number"))
                    plan["plan_source"] = "llm"
                    plan["legal_terms"] = []
                    return plan
            except Exception as e:
                logger.warning(f"LLM query planning failed: {e}")

        # 兜底：规则快速路径
        fallback = QueryRewriter._quick_parse(query)
        fallback["plan_source"] = "rule"
        fallback["intent"] = ""
        fallback["reasoning"] = "LLM 不可用，走规则兜底路径"
        return fallback

    @staticmethod
    def _quick_parse(query: str) -> Dict[str, Any]:
        """规则快速路径：正则+词典提取"""
        result = {
            "domain": "general",
            "law_name": "",
            "article_number": "",
            "article_index": 0,
            "legal_terms": [],
            "sub_queries": [query],
            "has_exact_ref": False,
            "search_strategy": "semantic",
        }

        # 提取法律名称
        m = re.search(r'《([^》]+)》', query)
        if m:
            name = m.group(1)
            if name in LAW_NAMES:
                result["law_name"] = LAW_NAMES[name]
            else:
                result["law_name"] = f"中华人民共和国{name}" if not name.startswith("中华人民共和国") else name
            result["has_exact_ref"] = True
        else:
            for short, full in sorted(LAW_NAMES.items(), key=lambda x: -len(x[0])):
                if short in query:
                    result["law_name"] = full
                    result["has_exact_ref"] = True
                    break

        # 提取条文
        m = re.search(r'第(\d+)条', query)
        if m:
            num = int(m.group(1))
            result["article_number"] = f"第{num}条"
            result["article_index"] = num
            result["has_exact_ref"] = True
        else:
            m = re.search(r'第([一二三四五六七八九十百千]+)条', query)
            if m:
                result["article_number"] = f"第{m.group(1)}条"
                result["has_exact_ref"] = True

        # 子查询
        if result["law_name"] and result["article_number"]:
            result["sub_queries"] = [query, f"{result['law_name']} {result['article_number']}"]
            result["search_strategy"] = "exact"
        elif result["law_name"]:
            result["sub_queries"] = [query, result["law_name"]]
            result["search_strategy"] = "hybrid"

        return result

    @staticmethod
    def _llm_plan(query: str, llm_func) -> Optional[Dict]:
        """调用 LLM 生成检索计划"""
        # 检查缓存
        cache_key = query.strip().lower()
        if cache_key in QueryRewriter._llm_cache:
            return QueryRewriter._llm_cache[cache_key]

        try:
            response = llm_func(QUERY_PLAN_PROMPT, query)
            # 解析 JSON
            plan = QueryRewriter._parse_llm_response(response)
            if plan:
                QueryRewriter._llm_cache[cache_key] = plan
                # 限制缓存大小
                if len(QueryRewriter._llm_cache) > 100:
                    QueryRewriter._llm_cache.clear()
            return plan
        except Exception:
            return None

    @staticmethod
    def _parse_llm_response(response: str) -> Optional[Dict]:
        """从 LLM 响应中解析 JSON"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试从 ```json 块中提取
        m = re.search(r'```(?:json)?\s*\n(.*?)\n```', response, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试从大括号中提取
        m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def number_to_chinese(num: int) -> str:
        """阿拉伯数字转中文数字"""
        cn_digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        if num < 10:
            return cn_digits[num]
        if num < 100:
            tens, ones = divmod(num, 10)
            result = cn_digits[tens] + "十"
            if ones:
                result += cn_digits[ones]
            return result
        return str(num)

    @staticmethod
    def chinese_to_int(cn: str) -> int:
        """中文数字转阿拉伯数字"""
        cn_digits = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
                     "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        cn_levels = {"十": 10, "百": 100, "千": 1000}
        result, temp = 0, 0
        for char in cn:
            if char in cn_digits:
                temp += cn_digits[char]
            elif char in cn_levels:
                temp = temp * cn_levels[char] if temp > 0 else cn_levels[char]
        result += temp
        return result if result > 0 else temp
