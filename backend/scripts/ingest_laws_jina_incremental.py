"""
法律条文入库脚本 - 增量版
跳过已入库的文件，只补未入库的部分
"""
import sys, os, re, logging, time
from pathlib import Path
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
BATCH_SIZE = 32

CATEGORY_MAP = {
    "民法典": "civil", "刑法": "criminal", "宪法": "constitutional",
    "行政法": "administrative", "经济法": "economic", "民法商法": "civil",
    "社会法": "social", "司法解释": "judicial", "诉讼与非诉讼程序法": "procedural",
    "部门规章": "regulatory", "DLC": "local", "案例": "case", "其他": "other",
    "宪法相关法": "constitutional", "行政法规": "administrative",
}

ARTICLE_RE   = re.compile(r'^(第[一二三四五六七八九十百零\d]+条)', re.MULTILINE)
TOTAL_CHUNKS = 148353

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
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL, "input": texts}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()["data"]
    data.sort(key=lambda x: x["index"])
    return [item["embedding"] for item in data]

def main():
    t0 = time.time()
    client = QdrantClient(host="localhost", port=6333)

    # 检查 collection 是否存在
    cols = [c.name for c in client.get_collections().collections]
    if COLLECTION not in cols:
        logger.info(f"Collection {COLLECTION} 不存在，请先运行完整版脚本")
        return

    # 获取已入库的 source_file 列表（用于跳过）
    existing_sources = set()
    offset = None
    while True:
        result = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=None,
            limit=1000,
            offset=offset,
            with_payload=["source_file"],
        )
        points, offset = result
        if not points:
            break
        for p in points:
            sf = p.payload and p.payload.get("source_file")
            if sf:
                existing_sources.add(sf)
        if offset is None:
            break

    logger.info(f"已有 {len(existing_sources)} 个文件在 Qdrant 中，将跳过")

    # 获取当前最大 point_id
    all_ids = client.search(
        collection_name=COLLECTION,
        query_vector=[0.0] * DIM,
        limit=1,
        search_params={"exact": True},
    )
    # 用 scroll 得到最大 id
    max_id = -1
    offset = None
    while True:
        result = client.scroll(collection_name=COLLECTION, limit=1000, offset=offset)
        points, offset = result
        if not points:
            break
        for p in points:
            if p.id > max_id:
                max_id = p.id
        if offset is None:
            break
    point_id = max_id + 1
    logger.info(f"从 point_id={point_id} 继续")

    # 扫描文件
    files = list(LAWS_DIR.rglob("*.md"))
    logger.info(f"文件总数: {len(files)}")

    # 过滤：只处理未入库的文件
    files_to_process = [fp for fp in files if str(fp) not in existing_sources]
    logger.info(f"需处理文件: {len(files_to_process)}（跳过 {len(files) - len(files_to_process)} 个已入库）")

    buf_texts  = []
    buf_pl      = []
    total_chunks = 0
    total_files  = 0
    errors       = 0

    for fi, fp in enumerate(files_to_process):
        chunks = chunk_file(fp)
        if not chunks:
            continue

        total_files += 1
        for ch in chunks:
            buf_texts.append(ch["content"])
            buf_pl.append(metadata(ch, fp))

        while len(buf_texts) >= BATCH_SIZE:
            try:
                vecs = embed_texts(buf_texts[:BATCH_SIZE])
                client.upsert(collection_name=COLLECTION, points=[
                    {"id": point_id + j, "vector": vecs[j], "payload": buf_pl[j]}
                    for j in range(BATCH_SIZE)
                ])
                point_id     += BATCH_SIZE
                total_chunks += BATCH_SIZE
                buf_texts    = buf_texts[BATCH_SIZE:]
                buf_pl       = buf_pl[BATCH_SIZE:]

                elapsed = time.time() - t0
                rate = total_chunks / elapsed if elapsed > 0 else 0
                remaining = len(files_to_process) - total_files
                eta = remaining * (elapsed / total_files) / 60 if total_files > 0 else 0
                logger.info(f"  [{total_chunks}] +{BATCH_SIZE} | {rate:.1f} chunks/s | ETA {eta:.0f}min | {total_files}/{len(files_to_process)} files")
            except Exception as e:
                logger.error(f"  upsert 失败: {e}")
                errors       += 1
                buf_texts    = buf_texts[BATCH_SIZE:]
                buf_pl       = buf_pl[BATCH_SIZE:]
                point_id    += BATCH_SIZE
                total_chunks += BATCH_SIZE

        if fi > 0 and fi % 50 == 0:
            logger.info(f"  [进度] {fi}/{len(files_to_process)} files, {total_chunks} new chunks")

    # 剩余数据
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

    info = client.get_collection(COLLECTION)
    elapsed = time.time() - t0
    logger.info("=" * 50)
    logger.info(f"完成! 耗时 {elapsed:.0f}s = {elapsed/60:.1f}min")
    logger.info(f"新增: {total_files} files, {total_chunks} chunks | Qdrant 总计: {info.points_count} | errors: {errors}")

if __name__ == "__main__":
    main()
