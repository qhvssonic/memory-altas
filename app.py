"""MemoryAtlas + DeepSeek 对话 Agent — 魔搭创空间部署入口

展示游戏引擎式记忆管理：预测性预加载、视锥剔除、LOD 精度切换、三层缓存、记忆簇。
"""

from __future__ import annotations

import os
import json
import re
import hashlib
import random
from pathlib import Path

import gradio as gr
from langchain_openai import ChatOpenAI

from memory_atlas.config import MemoryAtlasConfig
from memory_atlas.core.registry import Registry, MemoryRecord
from memory_atlas.core.tree_index import TreeIndex
from memory_atlas.storage.file_store import FileStore, MemoryChunk
from memory_atlas.scene.manager import SceneManager
from memory_atlas.core.cluster import ClusterManager


# === Embedder (hash-based, no model download needed) ===
class SimpleEmbedder:
    @property
    def dim(self) -> int:
        return 32

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
            rng = random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(self.dim)]
            norm = max(sum(x * x for x in vec) ** 0.5, 1e-9)
            results.append([x / norm for x in vec])
        return results


class DeepSeekLLMWrapper:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def complete(self, prompt, system="", **kw):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.llm.invoke(messages).content

    def complete_json(self, prompt, system="", **kw):
        raw = self.complete(prompt, system)
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if match:
                raw = match.group(1)
        try:
            return json.loads(raw)
        except Exception:
            return {}


# === Init ===
MODELSCOPE_TOKEN = os.environ.get("MODELSCOPE_API_KEY", "")

chat_llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3.2",
    api_key=MODELSCOPE_TOKEN or "sk-placeholder",
    base_url="https://api-inference.modelscope.cn/v1",
    temperature=0.7,
)

DATA_DIR = "/tmp/memory_atlas_data"
config = MemoryAtlasConfig(
    storage_path=DATA_DIR, hot_capacity=10, warm_capacity=50,
    prefetch_enabled=False, culling_enabled=True,
)
registry = Registry(Path(DATA_DIR) / "index.duckdb")
tree = TreeIndex(DATA_DIR)
file_store = FileStore(DATA_DIR)
embedder = SimpleEmbedder()
llm_wrapper = DeepSeekLLMWrapper(chat_llm)
scene = SceneManager(config, registry, tree, file_store, embedder, llm_wrapper)
cluster_mgr = ClusterManager(registry)
scene.initialize_session("chat_session")
_counter = [0]


def extract_entities_llm(text: str) -> list[str]:
    if not MODELSCOPE_TOKEN:
        words = re.findall(r"[a-zA-Z]{3,}", text)
        return list(set(w.lower() for w in words))[:5]
    try:
        resp = chat_llm.invoke([
            {"role": "system", "content": (
                "从用户消息中提取关键实体（技术概念、工具名、模块名等）。"
                "只返回 JSON 数组，例如 [\"jwt\", \"refresh token\"]。最多 5 个。"
            )},
            {"role": "user", "content": text},
        ])
        raw = resp.content.strip()
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if match:
                raw = match.group(1)
        entities = json.loads(raw)
        if isinstance(entities, list):
            return [str(e).lower().strip() for e in entities if e][:5]
    except Exception:
        pass
    words = re.findall(r"[a-zA-Z]{3,}", text)
    return list(set(w.lower() for w in words))[:5]


def ingest_turn(user_msg: str, assistant_msg: str):
    content = f"用户: {user_msg}\n助手: {assistant_msg}"
    if len(content.strip()) < 30:
        return
    _counter[0] += 1
    mid = f"chat_{_counter[0]:04d}"
    summary = content[:300]
    emb = embedder.embed([summary])[0]
    entities = extract_entities_llm(user_msg)

    registry.insert_memory(MemoryRecord(
        id=mid, session_id="chat", label=user_msg[:80],
        summary=summary, embedding=emb, importance_score=0.6,
    ))
    file_store.save_chunk(MemoryChunk(
        id=mid, session_id="chat", entities=entities,
        importance=0.6, title=user_msg[:80], content=content,
    ))
    for ent in entities:
        eid = registry.upsert_entity(ent, "concept")
        registry.link_memory_entity(mid, eid)
    cluster_mgr.auto_update_for_memory(mid, entities, threshold=3)


def chat(user_message: str, history: list[dict]):
    if not user_message.strip():
        return history, get_memory_status()
    if not MODELSCOPE_TOKEN:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "⚠️ 请设置 MODELSCOPE_API_KEY 环境变量"})
        return history, get_memory_status()

    memories = scene.get_memory_view(user_message)
    memory_context = ""
    if memories:
        lines = ["[以下是你的长期记忆，可以参考但不要逐条复述]"]
        for m in memories[:5]:
            lines.append(f"- [{m.lod}] {m.display_text[:150]}")
        lines.append("[记忆结束]")
        memory_context = "\n".join(lines)

    messages = [{"role": "system", "content": (
        "你是一个有长期记忆的 AI 助手。你能记住之前的对话内容。"
        "如果记忆中有相关信息，自然地融入回答中。不要说'根据我的记忆'。"
    )}]
    if memory_context:
        messages.append({"role": "system", "content": memory_context})
    for msg in history[-20:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    try:
        response = chat_llm.invoke(messages)
        assistant_msg = response.content
    except Exception as e:
        assistant_msg = f"调用出错: {e}"

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_msg})
    ingest_turn(user_message, assistant_msg)
    entities = extract_entities_llm(user_message)
    scene.update(user_message, current_entities=entities)
    return history, get_memory_status()


def get_memory_status() -> str:
    s = scene.stats()
    total = registry.count_memories()
    clusters = cluster_mgr.list_clusters()
    lines = [f"📊 总记忆: {total} | 热区: {s['hot']} | 温区: {s['warm']} | 冷区: {s['cold']} | 轮次: {s['turn_count']}"]
    lines.append("─" * 40)

    hot = scene.cache.get_hot()
    lines.append(f"\n🔴 热区 Hot ({len(hot)}):")
    if hot:
        for m in hot[:8]:
            lines.append(f"  [{m.lod}] {m.id}: {m.label[:50]}")
    else:
        lines.append("  (空 — 发起检索后记忆会进入热区)")

    warm = scene.cache.get_warm()
    lines.append(f"\n🟡 温区 Warm ({len(warm)}):")
    if warm:
        for m in warm[:8]:
            lines.append(f"  [{m.lod}] {m.id}: {m.label[:50]}")
    else:
        lines.append("  (空 — 话题转换后记忆从热区降级到这里)")

    all_mems = registry.list_memories(limit=20)
    lines.append(f"\n🔵 全部记忆 ({total}):")
    if all_mems:
        for m in all_mems:
            tier_icon = "🔴" if m.cache_tier == "hot" else "🟡" if m.cache_tier == "warm" else "⚪"
            lines.append(f"  {tier_icon} {m.id}: {m.label[:50]}")
            if m.summary and m.summary != m.label:
                lines.append(f"      摘要: {m.summary[:80]}")
    else:
        lines.append("  (空 — 开始对话后记忆会自动积累)")

    if clusters:
        lines.append(f"\n📦 记忆簇 ({len(clusters)}):")
        for c in clusters:
            lines.append(f"  {c.name} ({len(c.memory_ids)} 条) 标签: {c.entity_tags}")
    else:
        lines.append(f"\n📦 记忆簇: 暂无 (同一实体 ≥3 条记忆时自动创建)")
    return "\n".join(lines)


# === Gradio UI ===
with gr.Blocks(title="MemoryAtlas 记忆引擎") as demo:
    gr.Markdown("# 🧠 MemoryAtlas — 游戏引擎式 AI Agent 记忆管理")
    gr.Markdown("预测性预加载 · 视锥剔除 · LOD 精度切换 · 三层缓存 · 记忆簇自动归簇")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话", height=480)
            msg_input = gr.Textbox(label="输入消息", placeholder="试试下方的测试对话...", lines=2)
            with gr.Row():
                send_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清空对话")
        with gr.Column(scale=1):
            memory_status = gr.Textbox(label="🗄️ 记忆引擎状态", value=get_memory_status(), lines=30, interactive=False)

    state = gr.State([])

    def send(user_msg, history):
        history, status = chat(user_msg, history)
        return "", history, history, status

    def clear():
        scene.cache.clear()
        scene.initialize_session("new_session")
        return [], [], get_memory_status()

    msg_input.submit(send, [msg_input, state], [msg_input, chatbot, state, memory_status])
    send_btn.click(send, [msg_input, state], [msg_input, chatbot, state, memory_status])
    clear_btn.click(clear, [], [chatbot, state, memory_status])

    gr.Markdown("""
### 💡 测试对话（按顺序逐条发送，观察右侧记忆变化）

**第一阶段：聊 JWT 认证（观察记忆积累 + 自动归簇）**
```
我在做一个用户认证模块，打算用 JWT 方案，你觉得怎么样？
```
```
JWT 的 refresh token 应该怎么设计？我担心并发刷新会有竞态条件
```
```
我决定用滑动窗口策略来解决 refresh token 的竞态问题，token 有效期设为 7 天
```
```
JWT 密钥轮换怎么做比较好？我不想让用户重新登录
```

**第二阶段：话题转换（观察视锥剔除，JWT 记忆从热区降到温区）**
```
换个话题，我们来看看数据库性能优化的问题
```
```
users 表现在有 500 万行，按 email 查询特别慢，要 2 秒多
```

**第三阶段：回到 JWT（观察温区记忆快速恢复到热区）**
```
对了，之前说的 JWT 那个竞态问题，最后我们用的什么方案来着？
```
    """)

if __name__ == "__main__":
    demo.launch()
