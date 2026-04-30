"""
法律条文入库脚本 - Jina AI API (jina-embeddings-v3) + Qdrant
"""
import sys, os, re, logging, time, json
from pathlib import Path
import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

LAWS_DIR   = Path("/tmp/Laws")
COLLECTION = "law_copilot_laws"
DIM        = 1024
API_TOKEN  = "***REMOVED***"
MODEL      = "jina-embeddings-v3"
BATCH_SIZE = 32  # Jina API 单次最多32条

CATEGORY_MAP = {
    "民法典": "civil", "刑法": "criminal", "宪法": "constitutional",
    "行政法": "administrative", "经济法": "economic", "民法商法": "civil",
    "社会法": "social", "司法解释": "judicial", "诉讼与非诉讼程序法": "procedural",
    "部门规章": "regulatory", "DLC": "local", "案例": "case", "其他": "other",
    "宪法相关法": "constitutional", "行政法规": "administrative",
}

ARTICLE_RE   = re.compile(r'^(第[一二三四五六七八九十百零\d]+条)', re.MULTILINE)
PROGRESS_LOG = 50  # 每50个文件打印进度
TOTAL_CHUNKS = 148353  # 预估总量用于 ETA


def doc_type(fp: str) -> str:
    for k, v in CATEGORY_MAP.items():
        if k in fp:
            return v
    return "law"


def law_name(fp: str, text: str) -> str:
    m = re.search(r'《([^》]+)》', '\n'.join(text.split('\n')[:5]))
    return m.group(1) if m else Path(fp).stem


def chunk_file(fp: Path):
    try:
        text = fp.read_text(encoding='utf-8')
    except Exception:
        return []
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return [{"content": text.strip(), "art": "", "ch": "", "name": law_name(str(fp), text)}]
    chunks = []
    for i, m in enumerate(matches):
        seg = text[m.start(): matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
        if len(seg) < 10:
            continue
        chunks.append({"content": seg, "art": m.group(1), "ch": "", "name": law_name(str(fp), text)})
    return chunks


def metadata(chunk, fp):
    return {
        "doc_type": doc_type(str(fp)), "file_name": Path(fp).name,
        "source_file": str(fp), "article_number": chunk["art"],
        "chapter": chunk["ch"], "law_name": chunk["name"],
        "char_length": len(chunk["content"]),
    }


def embed_texts(texts: list[str]) -> list[list[float]]:
    """调用 Jina AI API 获取文本向量"""
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL, "input": texts}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()["data"]
    # 按 index 排序确保顺序一致
    data.sort(key=lambda x: x["index"])
    return [item["embedding"] for item in data]


def main():
    t0 = time.time()

    client = QdrantClient(host="localhost", port=6333)

    # 1. 删除旧 collection（如存在）
    cols = [c.name for c in client.get_collections().collections]
    if COLLECTION in cols:
        client.delete_collection(COLLECTION)
        logger.info(f"删除旧 collection: {COLLECTION}")
        time.sleep(1)

    # 2. 重建 collection
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
    )
    logger.info(f"创建 collection: {COLLECTION}, dim={DIM}")

    # 3. 扫描文件
    files = list(LAWS_DIR.rglob("*.md"))
    logger.info(f"文件总数: {len(files)}")

    point_id   = 0
    buf_texts  = []
    buf_pl     = []
    total_chunks = 0
    total_files  = 0
    errors = 0

    for fi, fp in enumerate(files):
        chunks = chunk_file(fp)
        if not chunks:
            continue

        total_files += 1
        for ch in chunks:
            buf_texts.append(ch["content"])
            buf_pl.append(metadata(ch, fp))

        # 凑满 BATCH_SIZE 就推送
        while len(buf_texts) >= BATCH_SIZE:
            try:
                t1 = time.time()
                vecs = embed_texts(buf_texts[:BATCH_SIZE])
                client.upsert(collection_name=COLLECTION, points=[
                    {"id": point_id + j, "vector": vecs[j], "payload": buf_pl[j]}
                    for j in range(BATCH_SIZE)
                ])
                point_id    += BATCH_SIZE
                total_chunks += BATCH_SIZE
                buf_texts   = buf_texts[BATCH_SIZE:]
                buf_pl      = buf_pl[BATCH_SIZE:]

                elapsed = time.time() - t0
                rate = total_chunks / elapsed if elapsed > 0 else 0
                eta  = (TOTAL_CHUNKS - total_chunks) / rate / 60 if rate > 0 else 0
                logger.info(f"  [{total_chunks}] +{BATCH_SIZE} | {rate:.1f} chunks/s | ETA {eta:.0f}min | {total_files}/{len(files)} files")
            except Exception as e:
                logger.error(f"  upsert 失败: {e}")
                errors += 1
                buf_texts  = buf_texts[BATCH_SIZE:]
                buf_pl     = buf_pl[BATCH_SIZE:]
                point_id  += BATCH_SIZE
                total_chunks += BATCH_SIZE

        # 进度记录
        if fi > 0 and fi % PROGRESS_LOG == 0:
            logger.info(f"  [进度] {fi}/{len(files)} files, {total_chunks} chunks, id={point_id}")

    # 3. 剩余数据
    if buf_texts:
        try:
            vecs = embed_texts(buf_texts)
            client.upsert(collection_name=COLLECTION, points=[
                {"id": point_id + j, "vector": vecs[j], "payload": buf_pl[j]}
                for j in range(len(buf_texts))
            ])
            total_chunks += len(buf_texts)
        except Exception as e:
            logger.error(f"  剩余 upsert 失败: {e}")

    # 4. 完成
    info = client.get_collection(COLLECTION)
    elapsed = time.time() - t0
    logger.info("=" * 50)
    logger.info(f"完成! 耗时 {elapsed:.0f}s = {elapsed/60:.1f}min")
    logger.info(f"总文件: {total_files}/{len(files)} | 总chunk: {total_chunks} | Qdrant: {info.points_count} | errors: {errors}")


if __name__ == "__main__":
    main()
