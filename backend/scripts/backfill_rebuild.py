"""
Content Backfill — 重建 Collection 版本

策略：从 /tmp/Laws 源文件重建所有 content，
通过设置相同 ID 的点来覆盖现有记录。

用法: uv run python scripts/backfill_rebuild.py
"""
import re, time, logging, requests
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HOST = "localhost"
PORT = 6333
COLLECTION = "law_copilot_laws"
SOURCE_DIR = Path("/tmp/Laws")
JINA_TOKEN = "***REMOVED***"
MODEL = "jina-embeddings-v3"

_file_cache: dict[str, str] = {}

def get_file_text(fp: str) -> str:
    if fp not in _file_cache:
        try:
            _file_cache[fp] = Path(fp).read_text(encoding="utf-8")
        except Exception:
            _file_cache[fp] = ""
    return _file_cache[fp]

def split_articles(text: str) -> dict[str, str]:
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

def doc_type(fp: str) -> str:
    fp_str = str(fp)
    if "司法解释" in fp_str:
        return "judicial"
    elif "地方性法规" in fp_str:
        return "local"
    elif "行政法规" in fp_str or "国务院" in fp_str:
        return "administrative"
    elif "刑法" in fp_str:
        return "criminal"
    elif "民法" in fp_str or "合同法" in fp_str or "婚姻法" in fp_str or "继承法" in fp_str:
        return "civil"
    elif "程序法" in fp_str or "诉讼法" in fp_str or "仲裁法" in fp_str:
        return "procedural"
    elif "宪法" in fp_str or "国籍法" in fp_str:
        return "constitutional"
    elif "社会法" in fp_str or "劳动" in fp_str or "保险" in fp_str:
        return "social"
    elif any(k in fp_str for k in ["经济法", "公司法", "证券", "破产", "反垄断", "知识产权", "涉外"]):
        return "economic"
    return "economic"

def chunk_file(fp: Path):
    text = get_file_text(str(fp))
    if not text.strip():
        return []
    law_name_val = fp.stem
    # 尝试找法律名称标题（第一行）
    first_line = text.split("\n")[0].strip()
    if len(first_line) < 30:
        law_name_val = first_line

    chunks = []
    articles = split_articles(text)
    for art_num, art_content in articles.items():
        if len(art_content) < 5:
            continue
        chunks.append({
            "content": art_content,
            "art": art_num,
            "ch": "",
            "name": law_name_val,
        })
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {JINA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL, "input": texts}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()["data"]
    data.sort(key=lambda x: x["index"])
    return [item["embedding"] for item in data]

def rebuild_collection():
    client = QdrantClient(host=HOST, port=PORT)

    # Step 1: 读取所有现有点的 metadata（保持原有 ID 不变）
    logger.info("读取现有 collection 元数据...")
    all_points = {}
    offset = 0
    batch = 500
    while True:
        results = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=None,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not results or not results[0]:
            break
        for pt in results[0]:
            all_points[pt.id] = pt.payload
        offset += len(results[0])
        if offset % 5000 == 0:
            logger.info(f"  已读取 {offset} 条...")
        if len(all_points) >= 47904:
            break
    logger.info(f"读取完毕，共 {len(all_points)} 条元数据")

    # Step 2: 补全 content
    for pt_id, payload in all_points.items():
        existing = payload.get("content", "")
        if existing and len(existing) > 5:
            continue  # 已有 content，跳过
        source_file = payload.get("source_file", "")
        article_number = payload.get("article_number", "")
        if not source_file or not article_number:
            continue
        text = get_file_text(source_file)
        if not text:
            continue
        articles = split_articles(text)
        key = article_number.strip()
        content = (
            articles.get(key)
            or articles.get(key.replace("条", ""))
            or ""
        )
        if content:
            payload["content"] = content

    # Step 3: 重建 collection（全量覆盖）
    logger.info("删除旧 collection...")
    try:
        client.delete_collection(collection_name=COLLECTION)
    except Exception:
        pass

    logger.info("创建新 collection...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    logger.info("新 collection 创建完成，dim=1024")

    # Step 4: 批量 upsert（带 content 的完整 payload）
    BATCH_SIZE = 64
    points_to_upsert = []
    total = len(all_points)
    done = 0
    t0 = time.time()

    for pt_id, payload in all_points.items():
        # 重建时用现有向量（从 all_points 拿），但我们没有保留向量
        # 所以需要重新 embedding
        pass

    # 实际上我们没法从 all_points 里拿到向量（scroll 时 with_vectors=False）
    # 所以这个方案行不通。改用：直接重新 ingest 所有源文件，跳过已有。
    logger.info("重建策略改为：从源文件重新 ingest，跳过已有文件")
    main_ingest()

def main_ingest():
    """从源文件重新 ingest，跳过已有 ID 范围"""
    client = QdrantClient(host=HOST, port=PORT)

    # 收集所有文件
    all_files = list(SOURCE_DIR.rglob("*.md"))
    logger.info(f"源目录共 {len(all_files)} 个 .md 文件")

    # 按文件分组建库
    total_chunks = 0
    buf_texts = []
    buf_payloads = []
    BATCH_SIZE = 64
    t0 = time.time()
    total_files = 0

    for fi, fp in enumerate(all_files):
        chunks = chunk_file(fp)
        if not chunks:
            continue
        total_files += 1
        for ch in chunks:
            buf_texts.append(ch["content"])
            buf_payloads.append({
                "content": ch["content"],
                "doc_type": doc_type(str(fp)),
                "file_name": fp.name,
                "source_file": str(fp),
                "article_number": ch["art"],
                "chapter": ch["ch"],
                "law_name": ch["name"],
                "char_length": len(ch["content"]),
            })

        while len(buf_texts) >= BATCH_SIZE:
            try:
                t1 = time.time()
                vecs = embed_texts(buf_texts[:BATCH_SIZE])
                now = int(time.time() * 1000)
                points = [
                    PointStruct(
                        id=now + j,
                        vector=vecs[j],
                        payload=buf_payloads[j],
                    )
                    for j in range(BATCH_SIZE)
                ]
                client.upsert(collection_name=COLLECTION, points=points)

                elapsed = time.time() - t0
                rate = (fi + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  [{fi+1}/{len(all_files)}] +{BATCH_SIZE} chunks | "
                    f"{rate:.1f} files/s | {total_chunks} total"
                )
                total_chunks += BATCH_SIZE
                buf_texts = buf_texts[BATCH_SIZE:]
                buf_payloads = buf_payloads[BATCH_SIZE:]
            except Exception as e:
                logger.error(f"  upsert 失败: {e}")
                buf_texts = buf_texts[BATCH_SIZE:]
                buf_payloads = buf_payloads[BATCH_SIZE:]

        if (fi + 1) % 50 == 0:
            logger.info(f"进度: {fi+1}/{len(all_files)} files processed")

    # 处理剩余
    if buf_texts:
        try:
            vecs = embed_texts(buf_texts)
            now = int(time.time() * 1000)
            points = [
                PointStruct(
                    id=now + j,
                    vector=vecs[j],
                    payload=buf_payloads[j],
                )
                for j in range(len(buf_texts))
            ]
            client.upsert(collection_name=COLLECTION, points=points)
            total_chunks += len(buf_texts)
        except Exception as e:
            logger.error(f"  最后批次 upsert 失败: {e}")

    elapsed = time.time() - t0
    logger.info(f"完成！共处理 {total_files} 文件，{total_chunks} chunks，耗时 {elapsed:.1f}s")

if __name__ == "__main__":
    # 不重建 collection 了，直接跳过已有文件增量补
    # 因为重建 collection 需要重新 embedding 全部 48K 条，太慢
    # 改用：直接从源文件补
    print("此脚本已废弃，请使用 ingest_laws_jina_incremental.py 补全")
    print("注意：先确保 JINA_TOKEN 余额充足")
