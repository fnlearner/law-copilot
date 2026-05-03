"""
QueryRewriter — 用户查询重写 + 结构化提取

从自然语言问题中提取:
  - 法律名称（如 "民事诉讼法"）
  - 条文编号（如 "第280条"）
  - 核心法律概念（如 "股权转让"、"违约责任"）

并生成多条子查询用于多路检索。
"""

import re
from typing import Dict, List, Optional, Tuple


# ===== 法律名称词典 =====
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
    "保险法": "中华人民共和国保险法",
    "票据法": "中华人民共和国票据法",
    "海商法": "中华人民共和国海商法",
    "招标投标法": "中华人民共和国招标投标法",
    "税收征收管理法": "中华人民共和国税收征收管理法",
    "土地管理法": "中华人民共和国土地管理法",
    "城市房地产管理法": "中华人民共和国城市房地产管理法",
    # 简称映射
    "合同法": "中华人民共和国民法典",  # 合同编归入民法典
    "劳动法": "中华人民共和国劳动法",
    "婚姻法": "中华人民共和国民法典",  # 婚姻家庭编
    "继承法": "中华人民共和国民法典",  # 继承编
}

# ===== 法律概念关键词 =====
LEGAL_CONCEPTS = [
    "股权转让", "股东出资", "公司回购", "公司清算", "公司解散",
    "合同效力", "合同解除", "违约责任", "损害赔偿", "违约金",
    "抵押", "质押", "担保", "保证",
    "债权", "债务", "破产", "重整",
    "商标侵权", "专利侵权", "著作权", "商业秘密",
    "劳动争议", "竞业限制", "经济补偿", "裁员",
    "内幕交易", "虚假陈述", "信息披露",
    "管辖权", "诉讼时效", "举证责任", "保全",
    "强制执行", "仲裁", "调解",
]

# ===== 主题 → 扩展搜索词 =====
TOPIC_EXPANSION = {
    # 装修/噪音
    "装修": ["装修", "噪音", "施工", "环境噪声污染防治法", "物业管理", "装修管理", "住宅室内装饰装修"],
    "噪音": ["噪音", "环境噪声", "装修", "施工", "环境噪声污染防治法"],
    "五一": ["节假日", "法定节假日", "装修", "噪音", "休息日"],
    "节假日装修": ["节假日", "装修", "噪音", "法定节假日", "禁止施工"],
    
    # 劳动/工资
    "加班": ["加班", "加班费", "劳动法", "工资支付"],
    "工资": ["工资", "工资支付", "劳动法"],
    "辞退": ["辞退", "解除劳动合同", "经济补偿", "劳动法"],
    "社保": ["社保", "社会保险", "五险一金"],
    
    # 房产/物业
    "租房": ["租房", "租赁", "房屋租赁", "合同法", "民法典"],
    "物业": ["物业", "物业管理", "物业费"],
    
    # 交通
    "交通事故": ["交通事故", "交通肇事", "赔偿", "侵权"],
    "违章": ["违章", "交通违章", "扣分", "罚款"],
    
    # 婚姻家庭
    "离婚": ["离婚", "婚姻", "民法典", "财产分割", "抚养权"],
    "继承": ["继承", "遗产", "遗嘱", "民法典"],
    
    # 合同
    "合同纠纷": ["合同纠纷", "违约", "合同解除", "民法典"],
    
    # 消费维权
    "退货": ["退货", "消费者权益", "三包", "消费者权益保护法"],
    "消费纠纷": ["消费", "消费者权益保护法", "三包", "赔偿"],
    
    # 行政
    "罚款": ["罚款", "行政处罚", "处罚"],
    "行政复议": ["行政复议", "行政诉讼", "复议"],
}

# ===== 查询重写 =====

class QueryRewriter:
    """
    用户查询重写器

    功能:
      1. 提取结构化检索条件 (law_name, article_number)
      2. 提取核心概念 (legal_terms)
      3. 生成多条子查询 (query_expansion)
    """

    @staticmethod
    def rewrite(query: str) -> Dict:
        """
        重写用户查询，返回结构化检索参数

        返回:
          {
            "original": 原始查询,
            "law_name": 提取的法律全称或 "",
            "article_number": 提取的条文编号或 "",
            "article_index": 数字条文号 (如 280) 或 0,
            "legal_terms": [核心法律概念列表],
            "sub_queries": [用于检索的多条子查询],
            "has_exact_ref": 是否包含精确法条引用,
          }
        """
        result = {
            "original": query,
            "law_name": "",
            "article_number": "",
            "article_index": 0,
            "legal_terms": [],
            "sub_queries": [query],
            "has_exact_ref": False,
        }

        # 1. 提取法律名称（先长匹配，再短匹配）
        law_name = QueryRewriter._extract_law_name(query)
        if law_name:
            result["law_name"] = law_name
            result["has_exact_ref"] = True

        # 2. 提取条文编号
        article_info = QueryRewriter._extract_article(query)
        if article_info:
            result["article_number"] = article_info[0]
            result["article_index"] = article_info[1]
            result["has_exact_ref"] = True

        # 3. 提取法律概念
        result["legal_terms"] = QueryRewriter._extract_legal_terms(query)

        # 4. 生成子查询
        result["sub_queries"] = QueryRewriter._expand_query(result)

        return result

    @staticmethod
    def _extract_law_name(query: str) -> str:
        """从查询中提取法律全称"""
        # 先尝试匹配带书名号的完整名称
        m = re.search(r'《([^》]+)》', query)
        if m:
            name = m.group(1)
            # 如果匹配到已知简称，转为全称
            if name in LAW_NAMES:
                return LAW_NAMES[name]
            return f"中华人民共和国{name}" if not name.startswith("中华人民共和国") else name

        # 匹配已知简称
        for short, full in sorted(LAW_NAMES.items(), key=lambda x: -len(x[0])):
            if short in query:
                return full

        return ""

    @staticmethod
    def _extract_article(query: str) -> Optional[Tuple[str, int]]:
        """从查询中提取条文编号"""
        # "第280条" / "第280条之一"
        m = re.search(r'第(\d+)条', query)
        if m:
            num = int(m.group(1))
            article_str = f"第{num}条"
            # 转中文
            cn = QueryRewriter._to_chinese_number(num)
            cn_article = f"第{cn}条"
            return cn_article, num

        # "第二百八十条" 中文数字
        m = re.search(r'第([一二三四五六七八九十百千]+)条', query)
        if m:
            cn_num = m.group(1)
            num = QueryRewriter._chinese_to_int(cn_num)
            return f"第{cn_num}条", num

        return None

    @staticmethod
    def _extract_legal_terms(query: str) -> List[str]:
        """提取核心法律概念（含主题扩展）"""
        found = []
        # 先长匹配 LEGAL_CONCEPTS
        for concept in sorted(LEGAL_CONCEPTS, key=len, reverse=True):
            if concept in query:
                found.append(concept)

        # 再匹配 TOPIC_EXPANSION 主题词
        for topic, expansions in sorted(TOPIC_EXPANSION.items(), key=lambda x: -len(x[0])):
            if topic in query:
                found.extend(expansions)

        return found[:6]

    @staticmethod
    def _expand_query(info: Dict) -> List[str]:
        """生成多条子查询"""
        queries = [info["original"]]
        law = info["law_name"]
        article = info["article_number"]

        if law and article:
            queries.append(f"{law} {article}")
            queries.append(f"{law} 第{info['article_index']}条")
        elif law:
            queries.append(f"{law}")
            for term in info["legal_terms"]:
                queries.append(f"{law} {term}")

        # 主题扩展子查询（针对无精确法条引用的问题）
        if not info["has_exact_ref"]:
            terms = info["legal_terms"]
            if terms:
                # 核心概念单独作为子查询
                for term in terms[:3]:
                    queries.append(f"{term}")
                # 组合查询
                if len(terms) >= 2:
                    queries.append(f"{' '.join(terms[:3])} 法律规定")

        return list(set(queries))

    @staticmethod
    def _to_chinese_number(num: int) -> str:
        """阿拉伯数字转中文数字"""
        cn_digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        cn_units = ["", "十", "百", "千"]
        if num < 10:
            return cn_digits[num]
        if num < 100:
            tens, ones = divmod(num, 10)
            result = cn_digits[tens] + "十"
            if ones:
                result += cn_digits[ones]
            return result
        if num < 1000:
            hundreds = num // 100
            rest = num % 100
            result = cn_digits[hundreds] + "百"
            if rest:
                if rest < 10:
                    result += "零" + cn_digits[rest]
                else:
                    result += QueryRewriter._to_chinese_number(rest)
            return result
        return str(num)

    @staticmethod
    def _chinese_to_int(cn: str) -> int:
        """中文数字转阿拉伯数字"""
        cn_digits = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
                     "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        cn_levels = {"十": 10, "百": 100, "千": 1000}
        result, temp = 0, 0
        for char in cn:
            if char in cn_digits:
                temp += cn_digits[char]
            elif char in cn_levels:
                if temp == 0:
                    temp = cn_levels[char]
                else:
                    temp *= cn_levels[char]
            else:
                result += temp
                temp = 0
        result += temp
        return result if result > 0 else temp
