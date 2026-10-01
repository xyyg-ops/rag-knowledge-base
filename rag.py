# -*- coding: utf-8 -*-
"""RAG 核心：检索 → 拼提示词 → 调对话模型 → 带引用回答。"""
import requests
import config
from embedder import Embedder
from store import VectorStore

PROMPT = """你是企业知识库助手。请严格依据下面的【参考资料】回答用户问题。

规则：
1. 只使用参考资料中的信息，不要编造。
2. 如果参考资料不足以回答，直接说"根据现有资料无法回答"，并说明缺什么。
3. 回答末尾用 [来源: 文件名] 标注引用的资料。

【参考资料】
{context}

【用户问题】
{question}

【回答】"""


class RAG:
    def __init__(self, top_k=None):
        self.embedder = Embedder()
        self.store = VectorStore.load()
        self.top_k = top_k or config.TOP_K

    def retrieve(self, question, top_k=None):
        qv = self.embedder.encode(question)[0]
        return self.store.search(qv, top_k=top_k or self.top_k)

    @staticmethod
    def build_context(hits):
        parts = []
        for i, h in enumerate(hits, 1):
            c = h["chunk"]
            section = (c.get("meta") or {}).get("section") or ""
            head = f"[{i}] 来源：{c['source']}" + (f" · {section}" if section else "")
            parts.append(f"{head}\n{c['text']}")
        return "\n\n---\n\n".join(parts)

    def _chat(self, prompt):
        if not config.CHAT_API_KEY:
            return None
        resp = requests.post(
            f"{config.CHAT_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.CHAT_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": config.CHAT_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.2},
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"对话接口返回 {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]

    def ask(self, question, top_k=None):
        hits = self.retrieve(question, top_k)
        context = self.build_context(hits)
        prompt = PROMPT.format(context=context, question=question)
        try:
            answer = self._chat(prompt)
        except Exception as e:
            answer = f"[调用对话模型失败] {e}"
        if answer is None:
            answer = ("（未配置 DEEPSEEK_API_KEY，仅展示检索结果。\n"
                      "  配置 Key 后此处会输出模型生成的带引用回答。）")
        return {"question": question, "answer": answer, "hits": hits,
                "context": context, "prompt": prompt}
