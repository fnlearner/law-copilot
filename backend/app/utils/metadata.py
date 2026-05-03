"""
元数据处理工具函数
"""


def build_title(meta: dict, fallback: str = "未知法律") -> str:
    """
    从 metadata 构建显示标题。

    优先级：
    1. law_name + article_number（组合格式，适用于搜索结果）
    2. law_name（仅有法律名称）
    3. file_name
    4. fallback

    Args:
        meta: 文档元数据字典
        fallback: 终极默认值

    Returns:
        格式化后的标题字符串
    """
    law_name = meta.get("law_name", "")
    article_number = meta.get("article_number", "")
    file_name = meta.get("file_name", "")

    # 优先：法律名 + 条号
    if law_name:
        title = f"{law_name} {article_number}".strip()
        if title:
            return title
        # 有 law_name 但 article_number 为空，仍用法律名
        return law_name

    # 其次：文件名
    if file_name:
        return file_name

    # 裁判文书（accusation）
    accusation = meta.get("accusation", "")
    if accusation:
        return f"{accusation}案"

    return fallback


def build_source_reference(meta: dict) -> dict:
    """
    从 metadata 构建统一格式的来源引用信息。

    Args:
        meta: 文档元数据字典

    Returns:
        包含 title, source, doc_type 的字典
    """
    return {
        "title": build_title(meta),
        "source": meta.get("source_file", ""),
        "doc_type": meta.get("doc_type", "unknown"),
    }
