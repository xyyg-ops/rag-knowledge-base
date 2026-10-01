# -*- coding: utf-8 -*-
"""FastAPI 接口服务 —— 对应简历里"使用 FastAPI 提供接口，含参数校验与错误处理"。

启动：
    uvicorn api:app --reload --port 8000
接口文档（给面试官截图的就是这个）：
    http://127.0.0.1:8000/docs
"""
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from rag import RAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("api.log", encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("rag-api")

app = FastAPI(title="企业文档知识库 RAG 服务", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    top_k: int = Field(default=4, ge=1, le=20, description="召回片段数")


class Hit(BaseModel):
    score: float
    source: str
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    hits: list[Hit]


_rag = None


def get_rag():
    global _rag
    if _rag is None:
        try:
            _rag = RAG()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _rag


@app.get("/health")
def health():
    return {"status": "healthy", "chat_model": config.CHAT_MODEL,
            "embed_model": config.EMBED_MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.time()
    try:
        out = get_rag().ask(req.question, top_k=req.top_k)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("问答失败：%s", e)
        raise HTTPException(status_code=500, detail=f"内部错误：{e}")
    log.info("Q=%s | 召回=%d | 耗时=%.2fs", req.question[:40], len(out["hits"]), time.time() - t0)
    return AskResponse(
        question=out["question"],
        answer=out["answer"],
        hits=[Hit(score=h["score"], source=h["chunk"]["source"], text=h["chunk"]["text"])
              for h in out["hits"]],
    )
