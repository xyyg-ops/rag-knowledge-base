# -*- coding: utf-8 -*-
"""切分策略 —— 这个文件就是你简历里那张"切分方案对比表"的来源。

三种策略：
  fixed    : 固定长度窗口（有重叠）
  para     : 按段落合并
  heading  : 按标题层级递归切分（最贴近企业文档的真实结构）
"""
import re
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    text: str
    source: str
    idx: int
    strategy: str
    meta: dict = None

    def to_dict(self):
        d = asdict(self)
        d["meta"] = self.meta or {}
        return d


def _window(text: str, size: int, overlap: int, source: str, strategy: str):
    chunks, start, i = [], 0, 0
    step = max(1, size - overlap)
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(Chunk(piece, source, i, strategy))
            i += 1
        start += step
    return chunks


def split_fixed(text, source, size=400, overlap=80, **kw):
    """固定长度窗口。最省事，但会把一句话从中间切断。"""
    return _window(text, size, overlap, source, "fixed")


def split_by_paragraph(text, source, size=400, overlap=80, **kw):
    """按空行切段，再把相邻小段合并到接近 size。段落语义更完整。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf, i = [], "", 0
    for p in paras:
        if len(buf) + len(p) + 2 <= size:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(Chunk(buf, source, i, "para"))
                i += 1
            # 单段本身就超长，退化成窗口切法
            if len(p) > size:
                for c in _window(p, size, overlap, source, "para"):
                    c.idx = i
                    chunks.append(c)
                    i += 1
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(Chunk(buf, source, i, "para"))
    return chunks


_HEADING = re.compile(r"^(#{1,6}\s+.+|第[一二三四五六七八九十]+[章节条].*|[一二三四五六七八九十]+、.*|\d+(\.\d+)*\s+.{0,40})$")


def split_by_heading(text, source, size=400, overlap=80, **kw):
    """按标题层级切分，保留章节归属。企业文档（制度/手册）效果最好。"""
    lines = text.split("\n")
    sections, cur_title, buf = [], "", []
    for ln in lines:
        if _HEADING.match(ln.strip()) and len(ln.strip()) < 60:
            if buf:
                sections.append((cur_title, "\n".join(buf).strip()))
            cur_title, buf = ln.strip(), []
        else:
            buf.append(ln)
    if buf:
        sections.append((cur_title, "\n".join(buf).strip()))

    chunks, i = [], 0
    for title, body in sections:
        body = (title + "\n" + body).strip() if title else body
        if not body:
            continue
        if len(body) <= size:
            chunks.append(Chunk(body, source, i, "heading", {"section": title}))
            i += 1
        else:
            for c in _window(body, size, overlap, source, "heading"):
                c.idx, c.meta = i, {"section": title}
                chunks.append(c)
                i += 1
    return chunks


STRATEGIES = {
    "fixed": split_fixed,
    "para": split_by_paragraph,
    "heading": split_by_heading,
}


def split(text, source, strategy="heading", size=400, overlap=80):
    if strategy not in STRATEGIES:
        raise ValueError(f"未知切分策略：{strategy}，可选 {list(STRATEGIES)}")
    return STRATEGIES[strategy](text, source, size=size, overlap=overlap)
