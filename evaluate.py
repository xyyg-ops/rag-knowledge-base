# -*- coding: utf-8 -*-
"""检索评测 —— 这是你简历里"检索命中率从 X% 提升到 Y%"的来源。

评测集格式（eval_set.json）：
[
  {"question": "年假有几天？", "answer_keywords": ["5天", "五天"], "source": "员工手册.md"},
  ...
]

命中判定：Top-K 召回的片段中，包含任一 answer_keywords 即算命中。
"""
import json
import sys
from pathlib import Path

import config
from embedder import Embedder
from store import VectorStore


def keyword_hit(hits, keywords):
    joined = "\n".join(h["chunk"]["text"] for h in hits)
    return any(k and k in joined for k in keywords)


def source_hit(hits, source):
    if not source:
        return True
    return any(h["chunk"]["source"] == source for h in hits)


def evaluate(eval_file=None, top_k=None, strategy_note=""):
    eval_file = Path(eval_file or (config.BASE_DIR / "eval_set.json"))
    if not eval_file.exists():
        print(f"评测集不存在：{eval_file}")
        return None

    cases = json.loads(eval_file.read_text(encoding="utf-8"))
    embedder = Embedder()
    store = VectorStore.load()
    top_k = top_k or config.TOP_K

    kw_ok = src_ok = 0
    rows = []
    for c in cases:
        qv = embedder.encode(c["question"])[0]
        hits = store.search(qv, top_k=top_k)
        k_ok = keyword_hit(hits, c.get("answer_keywords", []))
        s_ok = source_hit(hits, c.get("source"))
        kw_ok += k_ok
        src_ok += s_ok
        rows.append((c["question"], k_ok, s_ok,
                     hits[0]["score"] if hits else 0.0,
                     hits[0]["chunk"]["source"] if hits else "-"))

    n = len(cases)
    print("=" * 78)
    print(f"检索评测　切分策略：{strategy_note or '当前索引'}　Top-K={top_k}　用例数={n}")
    print("=" * 78)
    print(f"{'问题':<34}{'命中':<6}{'来源对':<7}{'最高分':<9}{'Top1来源'}")
    print("-" * 78)
    for q, k, s, sc, src in rows:
        print(f"{q[:32]:<34}{'✓' if k else '✗':<6}{'✓' if s else '✗':<7}{sc:<9.3f}{src}")
    print("-" * 78)
    print(f"内容命中率：{kw_ok}/{n} = {kw_ok / n * 100:.1f}%")
    print(f"来源命中率：{src_ok}/{n} = {src_ok / n * 100:.1f}%")
    return {"keyword_hit_rate": kw_ok / n, "source_hit_rate": src_ok / n, "n": n}


if __name__ == "__main__":
    evaluate(top_k=int(sys.argv[1]) if len(sys.argv) > 1 else None)
