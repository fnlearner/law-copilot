"""
Content Backfill 脚本
修复 ingest_laws_jina.py 漏了 content 字段的问题，
将 source_file + article_number 匹配的条文内容回填到 Qdrant payload

用法: uv run python scripts/backfill_content.py
"""
import re, time, logging
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION = "law_copilot_laws"
SOURCE_DIR = Path("/tmp/Laws")

# 按文件缓存，避免重复读文件
_file_cache: dict[str, str] = {}

def get_file_text(fp: str) -> str:
    if fp not in _file_cache:
        try:
            _file_cache[fp] = Path(fp).read_text(encoding="utf-8")
        except Exception:
            _file_cache[fp] = ""
    return _file_cache[fp]

def split_articles(text: str) -> dict[str, str]:
    """按'第X条'拆分文本，返回 {条文编号: 条文内容}"""
    pattern = r"(第[一二三四五六七八九十百零〇\d]+条)"
    parts = re.split(pattern, text)
    articles = {}
    i = 1
    while i < len(parts) - 1:
        art_num = parts[i].strip()
        art_content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if art_num and art_content:
            articles[art_num] = art_content
        i += 2
    return articles

def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    total = client.count(collection_name=COLLECTION, count_filter=None).count
    logger.info(f"总点数: {total}")

    updated = 0
    errors = 0
    skipped = 0
    offset = 0
    batch_size = 200

    while offset < total:
        results = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=None,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not results or not results[0]:
            break

        points = results[0]

        for pt in points:
            payload = pt.payload or {}
            existing = payload.get("content", "")

            # 已有有效 content，跳过
            if existing and len(existing) > 5:
                skipped += 1
                offset += 1
                continue

            source_file = payload.get("source_file", "")
            article_number = payload.get("article_number", "")

            if not source_file or not article_number:
                errors += 1
                offset += 1
                continue

            text = get_file_text(source_file)
            if not text:
                errors += 1
                offset += 1
                continue

            articles = split_articles(text)
            art_key = article_number.strip()
            content = (
                articles.get(art_key)
                or articles.get(art_key.replace("条", ""))
                or ""
            )

            if content:
                client.set_payload(
                    collection_name=COLLECTION,
                    payload={"content": content},
                    points=[pt.id],
                )
                updated += 1
            else:
                errors += 1

        offset += len(points)
        rate = updated / max(1, updated + errors) * 100
        logger.info(
            f"进度 {offset}/{total} | "
            f"已更新 {updated} | 跳过 {skipped} | 失败 {errors} | "
            f"成功率 {rate:.1f}%"
        )

    logger.info(f"完成！共更新 {updated} 条，失败 {errors} 条，跳过（已有）{skipped} 条")


if __name__ == "__main__":
    main()
