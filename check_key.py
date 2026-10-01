# -*- coding: utf-8 -*-
"""API Key 体检脚本 —— 填好 .env 后跑一次，确认两个 Key 都能用。

用法：
    python check_key.py

它会告诉你：
  1. 向量模型能不能调通、向量维度是多少
  2. 对话模型能不能调通
  3. 当前有哪些 embedding 模型可用
  4. 下一步该做什么
"""
import time

import requests

import config


def line(t):
    print("\n" + "=" * 66)
    print(t)
    print("=" * 66)


def mask(k):
    return f"{k[:6]}...{k[-4:]}（已脱敏）" if len(k) > 12 else "(太短，疑似复制不全)"


def check_embed():
    line("1. 向量模型 embedding")
    print(f"  平台地址 : {config.EMBED_BASE_URL}")
    print(f"  模型名称 : {config.EMBED_MODEL}")
    if not config.EMBED_API_KEY:
        print("  ❌ 未配置 Key —— 打开 .env，把 SILICONFLOW_API_KEY 填上")
        return None
    print(f"  API Key  : {mask(config.EMBED_API_KEY)}")

    t0 = time.time()
    try:
        r = requests.post(
            f"{config.EMBED_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {config.EMBED_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": config.EMBED_MODEL, "input": ["年假有几天？", "员工请假流程说明"]},
            timeout=60,
        )
    except Exception as e:
        print(f"  ❌ 网络请求失败：{e}")
        return None
    dt = time.time() - t0

    if r.status_code != 200:
        print(f"  ❌ HTTP {r.status_code}  耗时 {dt:.2f}s")
        print(f"     响应：{r.text[:400]}")
        hints = {
            401: "Key 无效或已过期 —— 检查是否复制完整（注意别漏掉 sk- 前缀）",
            403: "权限不足 —— 大概率是没完成实名认证，去 cloud.siliconflow.cn 做实名",
            404: "模型名不对 —— 跑本脚本第 3 步看可用模型列表，或去 pricing 页确认",
            429: "触发限流 —— 稍等一会儿再试",
        }
        print(f"     → {hints.get(r.status_code, '请对照上面的响应内容排查')}")
        return None

    data = r.json()["data"]
    dim = len(data[0]["embedding"])
    print(f"  ✅ 调用成功")
    print(f"     向量维度 : {dim}")
    print(f"     返回条数 : {len(data)}")
    print(f"     耗时     : {dt:.2f}s")
    return dim


def check_chat():
    line("2. 对话模型 chat")
    print(f"  平台地址 : {config.CHAT_BASE_URL}")
    print(f"  模型名称 : {config.CHAT_MODEL}")
    if not config.CHAT_API_KEY:
        print("  ⚠️  未配置 DEEPSEEK_API_KEY")
        print("     → 这不影响检索评测，但 RAG 的『生成回答』环节用不了")
        print("     → 想免费的话，也可以在硅基流动用 THUDM/GLM-Z1-9B-0414")
        return False
    print(f"  API Key  : {mask(config.CHAT_API_KEY)}")

    t0 = time.time()
    try:
        r = requests.post(
            f"{config.CHAT_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.CHAT_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": config.CHAT_MODEL,
                  "messages": [{"role": "user", "content": "只回复两个字：通了"}],
                  "temperature": 0},
            timeout=60,
        )
    except Exception as e:
        print(f"  ❌ 网络请求失败：{e}")
        return False
    dt = time.time() - t0

    if r.status_code != 200:
        print(f"  ❌ HTTP {r.status_code}  耗时 {dt:.2f}s")
        print(f"     响应：{r.text[:400]}")
        if r.status_code in (401, 402):
            print("     → 余额不足或 Key 无效，去 platform.deepseek.com 检查")
        return False

    msg = r.json()["choices"][0]["message"]["content"]
    print(f"  ✅ 调用成功  耗时 {dt:.2f}s")
    print(f"     模型回复：{msg.strip()[:40]}")
    return True


def list_models():
    line("3. 当前可用的 embedding 模型")
    if not config.EMBED_API_KEY:
        print("  （未配置 Key，跳过）")
        return
    try:
        r = requests.get(f"{config.EMBED_BASE_URL}/models",
                         headers={"Authorization": f"Bearer {config.EMBED_API_KEY}"},
                         timeout=30)
        if r.status_code != 200:
            print(f"  ❌ HTTP {r.status_code}：{r.text[:200]}")
            return
        ids = [m.get("id", "") for m in r.json().get("data", [])]
        emb = sorted(i for i in ids if any(k in i.lower()
                                           for k in ("embed", "bge", "gte", "m3")))
        print(f"  共 {len(ids)} 个模型，其中向量类 {len(emb)} 个：")
        for i in emb[:15]:
            mark = "  ← 当前使用" if i == config.EMBED_MODEL else ""
            print(f"    - {i}{mark}")
        if config.EMBED_MODEL not in ids:
            print(f"\n  ⚠️  你配置的 {config.EMBED_MODEL} 不在列表里，换个上面列出的")
    except Exception as e:
        print(f"  ❌ 获取失败：{e}")


def main():
    print("RAG 项目 · API Key 体检")
    dim = check_embed()
    chat_ok = check_chat()
    list_models()

    line("结论与下一步")
    if dim:
        print("  ✅ 向量模型可用 —— 可以建索引了：")
        print("       python main.py index")
        print("       python main.py eval")
        print("       python main.py compare      ← 这个会产出简历里的对比表")
        print("\n  ⚠️  换了向量模型必须重建索引，旧索引维度不同不能混用")
    else:
        print("  ❌ 向量模型不可用 —— 先解决上面标红的问题，再跑 main.py")
    if not chat_ok:
        print("\n  ℹ️  对话模型没配：检索和评测照常能跑，只是 /ask 不会生成回答")
    print()


if __name__ == "__main__":
    main()
