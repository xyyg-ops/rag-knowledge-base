# -*- coding: utf-8 -*-
"""全局配置。所有密钥从环境变量 / .env 读取，绝不写进代码。"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "index"

# ---------- 对话模型（DeepSeek，兼容 OpenAI 格式） ----------
CHAT_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "https://api.deepseek.com/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")

# ---------- 向量模型（智谱 embedding-3） ----------
EMBED_API_KEY = os.getenv("ZHIPU_API_KEY", "")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embedding-3")

# 本地兜底向量的维度（没有 API Key 时使用，方便离线开发）
LOCAL_EMBED_DIM = int(os.getenv("LOCAL_EMBED_DIM", "512"))

# ---------- 切分参数 ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "4"))
