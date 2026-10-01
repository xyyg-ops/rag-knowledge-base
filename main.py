# -*- coding: utf-8 -*-
"""命令行入口

用法：
    python main.py index                 # 建索引（默认 heading 切分）
    python main.py index --strategy para # 换切分策略重建
    python main.py ask "年假有几天？"     # 提问
    python main.py compare               # 三种切分策略横向对比（写进简历的东西）
    python main.py eval                  # 跑评测集
    python main.py rerank                # 对比 重排前 vs 重排后（写进简历的东西）
"""
import argparse
import shutil
import time

import config
from embedder import Embedder
from loader import load_dir
from splitter import split, STRATEGIES
from store import VectorStore


def build_index(strategy="heading", chunk_size=None, overlap=None, quiet=False):
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    docs = load_dir(config.DOCS_DIR)
    if not docs:
        raise SystemExit(f"docs 目录里没有可读文档：{config.DOCS_DIR}")

    chunks = []
    for text, meta in docs:
        for c in split(text, meta["source"], strategy=strategy, size=chunk_size, overlap=overlap):
            c.meta = {**(c.meta or {}), "path": meta["path"]}
            chunks.append(c)
    if not chunks:
        raise SystemExit("切分结果为空，检查文档内容和切分参数")

    emb = Embedder()
    t0 = time.time()
    vectors = emb.encode([c.text for c in chunks])
    dt = time.time() - t0

    vs = VectorStore()
    vs.add(vectors, [c.to_dict() for c in chunks])
    if config.INDEX_DIR.exists():
        shutil.rmtree(config.INDEX_DIR)
    vs.save()

    if not quiet:
        print(f"文档数：{len(docs)}　切片数：{len(chunks)}　切分策略：{strategy}"
              f"　(size={chunk_size}, overlap={overlap})")
        print(f"向量化模式：{emb.mode}　维度：{vs.dim}　耗时：{dt:.2f}s")
        print(f"索引已保存：{config.INDEX_DIR}")
    return {"docs": len(docs), "chunks": len(chunks), "dim": int(vs.dim),
            "mode": emb.mode, "seconds": dt}


def cmd_ask(question, top_k):
    from rag import RAG
    r = RAG(top_k=top_k)
    mode = (f"两阶段检索：向量宽召回(top_k×{config.RETRIEVE_MULTIPLIER}) + "
            f"{r.reranker.model} 重排") if r.reranker.active else "单阶段：仅向量召回"
    print(f"检索模式：{mode}")
    out = r.ask(question)
    print("=" * 70)
    print(f"问题：{out['question']}")
    print("=" * 70)
    print(out["answer"])
    print("-" * 70)
    print("检索到的片段：")
    for i, h in enumerate(out["hits"], 1):
        snippet = h["chunk"]["text"].replace("\n", " ")[:72]
        if "rerank_score" in h:
            score = (f"重排分 {h['rerank_score']:.4f}"
                     f"（向量分 {h.get('vector_score', 0):.3f}）")
        else:
            score = f"向量分 {h.get('score', 0):.3f}"
        print(f"  [{i}] {score}")
        print(f"       来源：{h['chunk']['source']}")
        print(f"       内容：{snippet}…")


def cmd_compare():
    """三种切分策略横向对比 —— 直接产出简历里的对比表"""
    from evaluate import run as eval_run
    rows = []
    for name in STRATEGIES:
        info = build_index(strategy=name, quiet=True)
        res = eval_run()
        rows.append((name, info["chunks"], res["keyword_hit_rate"], res["source_hit_rate"]))

    print("\n" + "=" * 74)
    print("切分策略横向对比")
    print("=" * 74)
    print(f"{'策略':<10}{'切片数':<10}{'内容命中率':<14}{'来源命中率'}")
    print("-" * 74)
    best = max(rows, key=lambda r: r[2])
    for n, c, k, s in rows:
        mark = "  ← 最优" if n == best[0] else ""
        print(f"{n:<10}{c:<10}{f'{k * 100:.1f}%':<12}{f'{s * 100:.1f}%'}{mark}")
    print("=" * 74)
    print(f"\n结论：{best[0]} 策略在当前语料上表现最好，内容命中率 {best[2] * 100:.1f}%。")
    print("（把这行填进简历，并把数字换成你自己的真实结果）")


def main():
    ap = argparse.ArgumentParser(description="企业文档知识库 RAG")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="建立索引")
    p_idx.add_argument("--strategy", default="heading", choices=list(STRATEGIES))
    p_idx.add_argument("--size", type=int)
    p_idx.add_argument("--overlap", type=int)

    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)

    sub.add_parser("compare", help="三种切分策略对比")
    sub.add_parser("eval", help="跑评测集")
    sub.add_parser("rerank", help="对比 重排前 vs 重排后")

    args = ap.parse_args()
    if args.cmd == "index":
        build_index(args.strategy, args.size, args.overlap)
    elif args.cmd == "ask":
        cmd_ask(args.question, args.top_k)
    elif args.cmd == "compare":
        cmd_compare()
    elif args.cmd == "eval":
        from evaluate import run
        run()
    elif args.cmd == "rerank":
        from evaluate import compare_rerank
        compare_rerank()


if __name__ == "__main__":
    main()
