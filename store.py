# -*- coding: utf-8 -*-
"""向量存储：numpy 矩阵 + json 元数据。

规模小的时候完全够用，也让你能清楚看到"向量检索"底层就是余弦相似度排序。
换成 Milvus / Chroma 时，只要替换 load / search 两个方法即可。
"""
import json
import numpy as np
import config


class VectorStore:
    def __init__(self, dim=None):
        self.dim = dim
        self.vectors = None      # (N, dim) float32，已归一化
        self.chunks = []         # list[dict]

    # ---------- 写入 ----------
    def add(self, vectors: np.ndarray, chunks: list):
        vectors = np.asarray(vectors, dtype=np.float32)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)
        self.dim = self.vectors.shape[1]

    # ---------- 检索 ----------
    def search(self, query_vec: np.ndarray, top_k=4, min_score=0.0):
        if self.vectors is None or not len(self.chunks):
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        sims = self.vectors @ q                       # 已归一化 → 点积即余弦相似度
        order = np.argsort(-sims)[:top_k]
        return [
            {"score": float(sims[i]), "chunk": self.chunks[i]}
            for i in order if float(sims[i]) >= min_score
        ]

    # ---------- 持久化 ----------
    def save(self, index_dir=None):
        index_dir = index_dir or config.INDEX_DIR
        index_dir.mkdir(parents=True, exist_ok=True)
        if self.vectors is not None:
            np.save(index_dir / "vectors.npy", self.vectors)
        (index_dir / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (index_dir / "meta.json").write_text(
            json.dumps({"dim": int(self.dim or 0), "count": len(self.chunks)}), encoding="utf-8"
        )
        return index_dir

    @classmethod
    def load(cls, index_dir=None):
        index_dir = index_dir or config.INDEX_DIR
        vs = cls()
        vec_file = index_dir / "vectors.npy"
        if not vec_file.exists():
            raise FileNotFoundError(f"索引不存在：{index_dir}，请先运行 python main.py index")
        vs.vectors = np.load(vec_file)
        vs.chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        vs.dim = vs.vectors.shape[1]
        return vs

    def __len__(self):
        return len(self.chunks)
