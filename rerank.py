# -*- coding: utf-8 -*-
"""重排（Rerank）—— 两阶段检索的第二阶段。

第一阶段：向量检索快速召回 top_k * N 个候选（召回的宽，保证不漏）
第二阶段：交叉编码器逐条精排（算的准，但慢，所以只用于少量候选）

这是工业界 RAG 的标准做法。向量检索是"双塔"结构，query 和 doc 各自编码后
算余弦相似度，快但精度有限；重排模型把 query 和 doc 拼在一起过一遍模型，
精度高很多，代价是只能处理少量候选。
"""
import requests

import config


class Reranker:
    def __init__(self, enabled=None):
        self.enabled = config.RERANK_ENABLED if enabled is None else enabled
        self.available = bool(config.EMBED_API_KEY)
        self.model = config.RERANK_MODEL

    @property
    def active(self):
        return bool(self.enabled and self.available)

    def rerank(self, query, hits, top_n):
        """对候选片段重排。

        失败时静默回退到原顺序 —— 重排是增强，不应该让整个链路挂掉。
        返回的每项会多一个 rerank_score 字段。
        """
        if not self.active or len(hits) <= 1:
            return hits[:top_n] if top_n else hits

        documents = [h["chunk"]["text"] for h in hits]
        try:
            resp = requests.post(
                f"{config.EMBED_BASE_URL}/rerank",
                headers={"Authorization": f"Bearer {config.EMBED_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": self.model, "query": query,
                      "documents": documents, "top_n": min(top_n or len(hits), len(documents))},
                timeout=60,
            )
            if resp.status_code != 200:
                print(f"  [警告] 重排接口 HTTP {resp.status_code}，回退到向量排序")
                return hits[:top_n] if top_n else hits
            results = resp.json().get("results", [])
        except Exception as e:
            print(f"  [警告] 重排失败（{e}），回退到向量排序")
            return hits[:top_n] if top_n else hits

        out = []
        for item in results:
            i = item.get("index")
            if i is None or i >= len(hits):
                continue
            h = dict(hits[i])
            h["rerank_score"] = float(item.get("relevance_score", 0.0))
            h["vector_score"] = h.get("score", 0.0)
            out.append(h)
        return out or (hits[:top_n] if top_n else hits)


if __name__ == "__main__":
    # 单独测试：python rerank.py
    r = Reranker()
    print(f"重排模型: {r.model}　启用: {r.enabled}　Key 可用: {r.available}　生效: {r.active}")
    fake = [{"score": 0.5, "chunk": {"text": t, "source": "test.md"}}
            for t in ["年假有 5 天", "报销需要发票", "考勤要打卡", "年假需提前 3 个工作日申请"]]
    for h in r.rerank("年假有几天？", fake, 3):
        print(f"  重排分 {h.get('rerank_score', 0):.4f}　{h['chunk']['text']}")
