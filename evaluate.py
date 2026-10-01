# -*- coding: utf-8 -*-
"""检索评测 —— 简历里量化指标的唯一来源。

跑法：
    python evaluate.py                 # 只测向量召回
    python main.py rerank              # 对比 重排前 vs 重排后
"""
import json
from pathlib import Path

import config
from embedder import Embedder
from rerank import Reranker
from store import VectorStore


def keyword_hit(hits, keywords):
    joined = "\n".join(h["chunk"]["text"] for h in hits)
    return any(k and k in joined for k in keywords)


def source_hit(hits, source):
    if not source:
        return True
    return any(h["chunk"]["source"] == source for h in hits)


def run(eval_file=None, top_k=None, use_rerank=False, strategy_note="", verbose=True):
    eval_file = Path(eval_file or (config.BASE_DIR / "eval_set.json"))
    if not eval_file.exists():
        print(f"评测集不存在：{eval_file}")
        return None

    cases = json.loads(eval_file.read_text(encoding="utf-8"))
    embedder = Embedder()
    store = VectorStore.load()
    reranker = Reranker(enabled=use_rerank)
    top_k = top_k or config.TOP_K
    # 有重排时先宽召回，再精排；没有重排就直接取 top_k
    fetch_k = top_k * config.RETRIEVE_MULTIPLIER if use_rerank else top_k

    kw_ok = src_ok = 0
    rows = []
    for c in cases:
        qv = embedder.encode(c["question"])[0]
        hits = store.search(qv, top_k=fetch_k)
        hits = reranker.rerank(c["question"], hits, top_n=top_k) if use_rerank else hits[:top_k]

        k_ok = keyword_hit(hits, c.get("answer_keywords", []))
        s_ok = source_hit(hits, c.get("source"))
        kw_ok += k_ok
        src_ok += s_ok
        top_score = (hits[0].get("rerank_score", hits[0].get("score", 0.0)) if hits else 0.0)
        rows.append((c["question"], k_ok, s_ok, top_score,
                     hits[0]["chunk"]["source"] if hits else "-"))

    n = len(cases)
    if verbose:
        tag = "重排后" if use_rerank else "仅向量召回"
        print("=" * 80)
        print(f"检索评测　{tag}　切分策略：{strategy_note or '当前索引'}"
              f"　Top-K={top_k}　候选={fetch_k}　用例数={n}")
        print("=" * 80)
        print(f"{'问题':<34}{'命中':<6}{'来源对':<7}{'最高分':<9}{'Top1来源'}")
        print("-" * 80)
        for q, k, s, sc, src in rows:
            print(f"{q[:32]:<34}{'✓' if k else '✗':<6}{'✓' if s else '✗':<7}{sc:<9.4f}{src}")
        print("-" * 80)
        print(f"内容命中率：{kw_ok}/{n} = {kw_ok / n * 100:.1f}%")
        print(f"来源命中率：{src_ok}/{n} = {src_ok / n * 100:.1f}%")
    return {"keyword_hit_rate": kw_ok / n, "source_hit_rate": src_ok / n,
            "n": n, "rows": rows}


def compare_rerank(top_k=None):
    """重排前 vs 重排后 —— 产出简历里"Top-1 准确率提升"的那条数据"""
    print("\n>>> 第一阶段：仅向量召回")
    before = run(top_k=top_k, use_rerank=False, verbose=False)
    print("\n>>> 第二阶段：向量召回 + 重排")
    after = run(top_k=top_k, use_rerank=True, verbose=False)

    if not before or not after:
        print("评测失败，检查索引是否存在（先跑 python main.py index）")
        return

    def diff(b, a):
        d = (a - b) * 100
        return f"{'+' if d >= 0 else ''}{d:.1f}%"

    print("\n" + "=" * 70)
    print("Rerank 效果对比")
    print("=" * 70)
    print(f"{'指标':<16}{'仅向量召回':<14}{'加重排':<14}{'变化'}")
    print("-" * 70)
    print(f"{'内容命中率':<16}{before['keyword_hit_rate'] * 100:<13.1f}%"
          f"{after['keyword_hit_rate'] * 100:<13.1f}%{diff(before['keyword_hit_rate'], after['keyword_hit_rate'])}")
    print(f"{'来源命中率(Top-1)':<16}{before['source_hit_rate'] * 100:<13.1f}%"
          f"{after['source_hit_rate'] * 100:<13.1f}%{diff(before['source_hit_rate'], after['source_hit_rate'])}")
    print("=" * 70)

    # 找出被重排救回来的案例
    b_map = {r[0]: r[2] for r in before["rows"]}
    fixed = [r[0] for r in after["rows"] if r[2] and not b_map.get(r[0])]
    if fixed:
        print(f"\n重排救回了 {len(fixed)} 条来源错误的案例：")
        for q in fixed:
            print(f"  ✓ {q}")
    broke = [r[0] for r in after["rows"] if not r[2] and b_map.get(r[0])]
    if broke:
        print(f"\n⚠️ 重排导致 {len(broke)} 条原本正确的案例出错（需要调 top_k 或候选数）：")
        for q in broke:
            print(f"  ✗ {q}")
    print("\n（把上面两个数字填进简历，写清楚语料规模与评测集大小）")


if __name__ == "__main__":
    import sys
    if "--compare" in sys.argv:
        compare_rerank()
    else:
        run(use_rerank="--rerank" in sys.argv,
            top_k=int(sys.argv[1]) if sys.argv[1].isdigit() else None)
