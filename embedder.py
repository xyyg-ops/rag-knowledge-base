# -*- coding: utf-8 -*-
"""向量化：优先调用智谱 embedding-3；没有 Key 时自动降级为本地字符 n-gram 向量。

降级的意义：让你在没有 API Key 的情况下也能把整条流水线跑通、做检索评测。
"""
import hashlib
import re
import numpy as np
import config


def _hash_embed(texts, dim=None):
    """本地兜底：字符二元组哈希到固定维度，再 L2 归一化。

    没有语义理解能力，但对小规模中文语料的关键词级检索够用，
    适合在没配 Key 时验证流程。
    """
    dim = dim or config.LOCAL_EMBED_DIM
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        t = re.sub(r"\s+", "", str(t))
        grams = [t[j:j + 2] for j in range(max(len(t) - 1, 1))] or [t]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
            vecs[i, h % dim] += 1.0
        n = float(np.linalg.norm(vecs[i]))
        if n > 0:
            vecs[i] /= n
    return vecs


def _api_embed(texts, batch=32):
    import requests
    out = []
    for s in range(0, len(texts), batch):
        part = [t[:2000] for t in texts[s:s + batch]]
        resp = requests.post(
            f"{config.EMBED_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {config.EMBED_API_KEY}"},
            json={"model": config.EMBED_MODEL, "input": part},
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"embedding 接口返回 {resp.status_code}: {resp.text[:300]}")
        data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
        out.extend(d["embedding"] for d in data)
    return np.array(out, dtype=np.float32)


class Embedder:
    def __init__(self, force_local=False):
        self.mode = "local" if (force_local or not config.EMBED_API_KEY) else "api"

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if t and t.strip() else "空" for t in texts]
        if self.mode == "api":
            try:
                v = _api_embed(texts)
                norms = np.linalg.norm(v, axis=1, keepdims=True)
                return v / np.clip(norms, 1e-8, None)
            except Exception as e:
                print(f"  [警告] 远程向量化失败，降级为本地向量：{e}")
                self.mode = "local"
        return _hash_embed(texts)


if __name__ == "__main__":
    from config import EMBED_API_KEY
    e = Embedder()
    print("向量化模式：", e.mode, "（未配置 ZHIPU_API_KEY 时为本地兜底）" if not EMBED_API_KEY else "")
    v = e.encode(["公司年假怎么申请？", "员工请假流程说明"])
    print("向量维度：", v.shape)
