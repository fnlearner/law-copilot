"""
裁判文书搜索服务 —— 基于 SQLite FTS5 BM25 全文检索

用于 CAIL2018 裁判文书（1,710,856 条）的快速关键词搜索。
供后端 API 的 search 和 chat 路由调用。
"""

import sqlite3
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class JudgmentSearchService:
    """裁判文书 BM25 搜索服务"""

    def __init__(self, db_path: str = "/tmp/cail2018_fts.db"):
        self.db_path = db_path
        self._ready = False

    async def initialize(self):
        """初始化：验证数据库可用"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.execute(
                "SELECT COUNT(*) FROM judgments_fts"
            )
            count = cur.fetchone()[0]
            conn.close()
            self._ready = True
            logger.info(
                f"JudgmentSearchService 就绪: {count} 条裁判文书索引"
            )
        except Exception as e:
            logger.warning(f"裁判文书数据库不可用: {e}")
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def search(
        self,
        keywords: list[str],
        limit: int = 20,
    ) -> list[dict]:
        """
        多关键词 BM25 搜索。

        Args:
            keywords: LLM 扩展后的关键词列表
            limit: 返回条数

        Returns:
            按 BM25 匹配度排序的结果列表
        """
        if not keywords or not self._ready:
            return []

        # 清理 FTS5 特殊字符：只保留中文、字母数字和空格，防止语法错误
        import re
        fts_keywords = [
            re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', kw).strip()
            for kw in keywords
        ]
        fts_keywords = [kw for kw in fts_keywords if kw]
        if not fts_keywords:
            return []

        fts_query = " OR ".join(fts_keywords)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                """
                SELECT j.id, j.full_text, j.accusation, j.relevant_articles,
                       j.criminals, j.imprisonment, j.life_imprisonment,
                       j.death_penalty, j.punish_of_money, rank
                FROM judgments_fts
                JOIN judgments j ON judgments_fts.rowid = j.id
                WHERE judgments_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            )
            results = [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

        return results

    def format_result(self, row: dict) -> dict:
        """将搜索行转为统一格式供 API 返回"""
        fact = row.get("full_text", "")
        return {
            "doc_type": "case",
            "title": f"{row.get('accusation', '未知')}案",
            "content": fact,
            "source": "cail2018",
            "relevance_score": -row.get("rank", 0),  # FTS5 rank 是负数，取正
            "metadata": {
                "accusation": row.get("accusation", ""),
                "relevant_articles": row.get("relevant_articles", ""),
                "criminals": row.get("criminals", ""),
                "imprisonment": row.get("imprisonment"),
                "life_imprisonment": bool(row.get("life_imprisonment")),
                "death_penalty": bool(row.get("death_penalty")),
                "punish_of_money": row.get("punish_of_money"),
                "source_type": "cail2018",
            },
        }

    async def shutdown(self):
        """关闭时清理"""
        self._ready = False
        logger.info("JudgmentSearchService 已关闭")
