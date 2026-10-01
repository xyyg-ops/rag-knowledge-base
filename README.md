# 企业文档知识库 RAG 智能问答系统

基于 Python 实现的检索增强生成（RAG）系统，支持 PDF / Word / txt / Markdown 文档的解析、
清洗、切分、向量检索与带引用的问答，并提供 FastAPI 接口服务与检索评测脚本。

## 功能

| 模块 | 文件 | 说明 |
|---|---|---|
| 文档解析与清洗 | `loader.py` | PDF / docx / txt / md，含表格提取与噪声行过滤 |
| 切分策略 | `splitter.py` | 固定窗口 / 按段落 / 按标题递归，三种可对比 |
| 向量化 | `embedder.py` | 智谱 embedding-3；无 Key 时自动降级为本地向量 |
| 向量存储与检索 | `store.py` | numpy 余弦相似度检索，可平滑替换为 Milvus / Chroma |
| RAG 问答 | `rag.py` | 检索 → 拼提示词 → 生成，强制引用来源，拒答未知问题 |
| 接口服务 | `api.py` | FastAPI，含参数校验、错误处理与日志 |
| 检索评测 | `evaluate.py` | 命中率评测，产出可写进简历的量化指标 |

## 快速开始

```bash
pip install -r requirements.txt

cp .env.example .env        # 填入 DEEPSEEK_API_KEY 和 ZHIPU_API_KEY

python main.py index                    # 建索引（默认按标题切分）
python main.py ask "年假有几天？"         # 提问
python main.py eval                     # 跑评测集，输出命中率
python main.py compare                  # 三种切分策略横向对比
```

启动接口服务：

```bash
uvicorn api:app --reload --port 8000
# 打开 http://127.0.0.1:8000/docs
```

## 项目结构

```
rag-kb/
├── config.py        全局配置（密钥从 .env 读取）
├── loader.py        文档解析与清洗
├── splitter.py      三种切分策略
├── embedder.py      向量化（含本地兜底）
├── store.py         向量存储与余弦检索
├── rag.py           检索 + 生成
├── evaluate.py      检索评测
├── main.py          命令行入口
├── api.py           FastAPI 服务
├── eval_set.json    评测集（30 条）
└── docs/            示例文档
```

## 设计说明

**为什么自己做向量存储而不是直接用 LangChain？**
语料规模在万级以内时，`numpy` 矩阵加余弦相似度就足够，而且能清楚看到检索的底层过程。
`store.py` 的 `load` / `search` 两个方法做了接口隔离，换成 Milvus 只需要改这两个方法。

**为什么 prompt 里强制要求引用来源？**
企业知识库场景下，无出处的回答等于没有价值。系统被要求"资料不足时直接说无法回答"，
这是抑制大模型幻觉最直接的手段。

**为什么做了三种切分策略？**
切分粒度直接决定检索命中率。切片过大噪声多，过小则语义不完整。
项目通过 `python main.py compare` 在同一评测集上横向对比三种策略，用数据而非感觉选择方案。

## 评测结果

在自建 30 条中文问答评测集上（内容命中率 = Top-K 召回片段中包含答案关键词的比例）：

| 切分策略 | 切片数 | 内容命中率 | 来源命中率 |
|---|---|---|---|
| fixed | 待填 | 待填 | 待填 |
| para | 待填 | 待填 | 待填 |
| heading | 待填 | 待填 | 待填 |

> 运行 `python main.py compare` 生成真实数据后填写本表。

> ⚠️ **重要**：必须配置 `ZHIPU_API_KEY` 后再评测。未配置时系统会降级为本地字符
> n-gram 向量，它在关键词匹配上几乎必然满分（本项目实测三种策略均接近 100%），
> 这个数字**不能写进简历**——面试官一问"为什么固定窗口切分反而最好"就答不上来。
> 用真实 embedding 模型跑出来的数字才有意义，通常 heading 策略在结构化文档上最优。
> 同时把 `docs/` 换成你自己真实的语料（几十份文档），2 份文档的评测结果没有说服力。

## 待改进

- 检索结果未做 rerank，可接入 bge-reranker 提升 Top-1 准确率
- 未做多轮对话的上下文压缩，长会话会超出上下文窗口
- 向量库为单机 numpy 实现，未做分布式与增量更新
