"""
法律条文入库脚本 - 增量版（基于 vector_sdk）

跳过已入库的文件，只补未入库的部分。
依赖：backend/vector_sdk.py
"""
import sys, os, re, logging, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from vector_sdk import LawVectorSDK, LawChunk, VectorPoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

LAWS_DIR   = Path("/tmp/Laws")
COLLECTION = "law_copilot_laws"

CATEGORY_MAP = {
    "民法典": "civil", "刑法": "criminal", "宪法": "constitutional",
    "行政法": "administrative", "经济法": "economic", "民法商法": "civil",
    "社会法": "social", "司法解释": "judicial", "诉讼与非诉讼程序法": "procedural",
    "部门规章": "regulatory", "DLC": "local", "案例": "case", "其他": "other",
    "宪法相关法": "constitutional", "行政法规": "administrative",
}

ARTICLE_RE = re.compile(r'^(第[一二三四五六七八九十百零\d]+条)', re.MULTILINE)

# ─── 分块 & metadata（保留原有领域逻辑不动）──────────────────────────────────────

def doc_type(fp: str) -> str:
    for k, v in CATEGORY_MAP.items():
        if k in fp:
            return v
    return "law"

def law_name(fp: str, text: str) -> str:
    m = re.search(r'《([^》]+)》', '\n'.join(text.split('\n')[:5]))
    return m.group(1) if m else Path(fp).stem

def chunk_file(fp: Path):
    """按第X条切分，返回 chunk 列表。"""
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

def make_chunk_id(source_file: str, article_number: str, index: int) -> str:
    """生成稳定可重入的 chunk ID，用于 upsert 去重。"""
    sf_hash = str(hash(source_file))[-8:]
    return f"{sf_hash}_{article_number}_{index}"


# ─── 主流程 ─────────────────────────────────────────────────────────────────

def main():
    from dotenv import load_dotenv
    load_dotenv('.env')

    t0 = time.time()

    # 初始化 SDK（自动处理 collection 建表）
    sdk = LawVectorSDK(collection=COLLECTION, host="localhost", port=6333)

    # 增量：读取已有 chunk 的 source_file，跳过已入库文件
    existing_sources = _load_existing_sources(sdk)
    logger.info(f"已有 {len(existing_sources)} 个文件在 Qdrant 中，将跳过")

    # 扫描文件
    all_files = list(LAWS_DIR.rglob("*.md"))
    files_to_process = [fp for fp in all_files if str(fp) not in existing_sources]
    logger.info(f"文件总数: {len(all_files)}，需处理: {len(files_to_process)}（跳过 {len(all_files) - len(files_to_process)} 个已入库）")

    if not files_to_process:
        logger.info("没有新文件需要处理，退出")
        return

    # 批量 ingest
    buf_chunks: list[LawChunk] = []
    BATCH_SIZE = 64
    total_chunks = 0
    total_files  = 0
    errors       = 0

    for fi, fp in enumerate(files_to_process):
        file_chunks = chunk_file(fp)
        if not file_chunks:
            continue

        total_files += 1
        for idx, ch in enumerate(file_chunks):
            buf_chunks.append(LawChunk(
                content=ch["content"],
                chunk_id=make_chunk_id(str(fp), ch["art"], idx),
                metadata={
                    "doc_type": doc_type(str(fp)),
                    "file_name": fp.name,
                    "source_file": str(fp),
                    "article_number": ch["art"],
                    "chapter": ch["ch"],
                    "law_name": ch["name"],
                    "char_length": len(ch["content"]),
                },
            ))

        # 凑够一批就 ingest
        while len(buf_chunks) >= BATCH_SIZE:
            try:
                sdk.store.upsert(_chunks_to_points(buf_chunks[:BATCH_SIZE]))
                total_chunks += BATCH_SIZE
                buf_chunks = buf_chunks[BATCH_SIZE:]

                elapsed = time.time() - t0
                rate = total_chunks / elapsed if elapsed > 0 else 0
                eta = (len(files_to_process) - total_files) * (elapsed / total_files) / 60 if total_files > 0 else 0
                logger.info(f"  [{total_chunks}] +{BATCH_SIZE} | {rate:.1f} chunks/s | ETA {eta:.0f}min | {total_files}/{len(files_to_process)} files")
            except Exception as e:
                logger.error(f"  upsert 失败: {e}")
                errors += 1
                buf_chunks = buf_chunks[BATCH_SIZE:]
                total_chunks += BATCH_SIZE

        if fi > 0 and fi % 50 == 0:
            logger.info(f"  [进度] {fi}/{len(files_to_process)} files, {total_chunks} new chunks")

    # 剩余不足一批的
    if buf_chunks:
        try:
            sdk.store.upsert(_chunks_to_points(buf_chunks))
            total_chunks += len(buf_chunks)
        except Exception as e:
            logger.error(f"  剩余 upsert 失败: {e}")
            errors += 1

    elapsed = time.time() - t0
    logger.info("=" * 50)
    logger.info(f"完成! 耗时 {elapsed:.0f}s = {elapsed/60:.1f}min")
    logger.info(f"新增: {total_files} files, {total_chunks} chunks | Qdrant 总计: {sdk.count()} | errors: {errors}")


def _load_existing_sources(sdk: LawVectorSDK) -> set[str]:
    """遍历 Qdrant，收集已有 chunk 的 source_file 集合。"""
    existing: set[str] = set()
    offset = None
    while True:
        hits, offset = sdk.store.scroll(limit=1000, offset=offset)
        for h in hits:
            sf = h.payload.get("source_file")
            if sf:
                existing.add(sf)
        if offset is None:
            break
    return existing


def _chunks_to_points(chunks: list[LawChunk]) -> list[VectorPoint]:
    """把 LawChunk 批量向量化并包装成 VectorPoint。"""
    from vector_sdk import JinaEmbeddingProvider
    provider = JinaEmbeddingProvider()
    texts = [c.content for c in chunks]
    vectors = provider.embed(texts)
    return [
        VectorPoint(
            id=c.as_point_id(),
            vector=vec,
            payload={
                "content": c.content,
                "chunk_id": c.chunk_id,
                "metadata": c.metadata,
            },
        )
        for c, vec in zip(chunks, vectors)
    ]


if __name__ == "__main__":
    main()
