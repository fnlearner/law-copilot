#!/usr/bin/env python
"""
RAG 搜索测试脚本
直接用 Qdrant 原生客户端 + Jina API 查向量库
"""
import requests
import numpy as np
from qdrant_client import QdrantClient

# ── 配置 ──
API_TOKEN = "***REMOVED***"
MODEL = "jina-embeddings-v3"
COLLECTION = "law_copilot_laws"
TOP_K = 5


def embed_query(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"},
        json={"model": MODEL, "input": texts},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    data.sort(key=lambda x: x["index"])
    return [item["embedding"] for item in data]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def search(query: str, top_k: int = TOP_K):
    # 1. 查询向量化
    query_vec = embed_query([query])[0]

    # 2. Qdrant 搜索
    client = QdrantClient(host="localhost", port=6333)
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=top_k,
    )

    # 3. 打印结果
    print(f"\n{'='*60}")
    print(f"查询: {query}")
    print(f"{'='*60}")

    for i, r in enumerate(results, 1):
        payload = r.payload
        score = r.score
        content = payload.get("content", "")[:200]
        law_name = payload.get("law_name", "")
        article = payload.get("article_number", "")
        doc_type = payload.get("doc_type", "")

        print(f"\n【{i}】{law_name} {article}")
        print(f"    相关度: {score:.4f} | 类型: {doc_type}")
        print(f"    内容: {content}...")
        print()


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "劳动合同解除的赔偿标准"
    search(query)
