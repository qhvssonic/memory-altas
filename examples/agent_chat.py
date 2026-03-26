"""真实对话 Agent + MemoryAtlas 记忆引擎 + Gradio 界面

使用 DeepSeek 作为 LLM，MemoryAtlas 提供长期记忆。
每轮对话自动：检索相关记忆 → 注入上下文 → 对话 → 评估写入 → 场景管理更新。

运行前设置环境变量:
  set DEEPSEEK_API_KEY=your_key_here

运行:
  uv run python examples/agent_chat.py
"""

from __future__ import annotations

import os
import sys
import hashlib
import random
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gradio as gr
from langchain_openai import ChatOpenAI

from memory_atlas.config import MemoryAtlasConfig
from memory_atlas.core.registry import Registry, MemoryRecord
from memory_atlas.core.tree_index import TreeIndex
from memory_atlas.storage.file_store import FileStore, MemoryChunk
from memory_atlas.storage.cache import CachedMemory
from memory_atlas.scene.manager import SceneManager
from memory_atlas.core.cluster import ClusterManager
from memory_atlas.ingestion.chunker import Chunker


# === Embedder (本地 hash，不需要额外模型) ===
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


# === LLM wrapper for scene manager internal calls ===
class DeepSeekLLMWrapper:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def complete(self, prompt, system="", **kw):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.llm.invoke(messages)
        return resp.content

    def complete_json(self, prompt, system="", **kw):
        import json, re
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
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    print("⚠️  请设置 DEEPSEEK_API_KEY 环境变量")
    print("   Windows: set DEEPSEEK_API_KEY=sk-xxx")
    print("   Linux:   export DEEPSEEK_API_KEY=sk-xxx")
    sys.exit(1)

# DeepSeek LLM
chat_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.7,
)

# MemoryAtlas
TMP_DIR = tempfile.mkdtemp()
config = MemoryAtlasConfig(
    storage_path=TMP_DIR, hot_capacity=10, warm_capacity=50,
    prefetch_enabled=False, culling_enabled=True,
)
registry = Registry(Path(TMP_DIR) / "index.duckdb")
tree = TreeIndex(TMP_DIR)
file_store = FileStore(TMP_DIR)
embedder = SimpleEmbedder()
llm_wrapper = DeepSeekLLMWrapper(chat_llm)
scene = SceneManager(config, registry, tree, file_store, embedder, llm_wrapper)
cluster_mgr = ClusterManager(registry)
chunker = Chunker(strategy="paragraph")

scene.initialize_session("chat_session")
_counter = [0]


# === Core chat function ===

def ingest_turn(user_msg: str, assistant_msg: str):
    """将一轮对话写入记忆。"""
    content = f"用户: {user_msg}\n助手: {assistant_msg}"
    if len(content.strip()) < 30:
        return

    _counter[0] += 1
    mid = f"chat_{_counter[0]:04d}"
    summary = content[:300]
    emb = embedder.embed([summary])[0]

    # 简单实体提取：取用户消息中的关键词
    import re
    words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]{3,}", user_msg)
    entities = list(set(w.lower() for w in words if len(w) >= 2))[:5]

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
    """主对话函数：检索记忆 → 构建 prompt → 调用 DeepSeek → 写入记忆 → 更新场景。"""
    if not user_message.strip():
        return history, get_memory_status()

    # 1. 检索相关记忆
    memories = scene.get_memory_view(user_message)
    memory_context = ""
    if memories:
        lines = ["[以下是你的长期记忆，可以参考但不要逐条复述]"]
        for m in memories[:5]:
            lines.append(f"- [{m.lod}] {m.display_text[:150]}")
        lines.append("[记忆结束]")
        memory_context = "\n".join(lines)

    # 2. 构建消息
    messages = [{"role": "system", "content": (
        "你是一个有长期记忆的 AI 助手。你能记住之前的对话内容。"
        "如果记忆中有相关信息，自然地融入回答中。"
        "不要说'根据我的记忆'这种话，直接用就好。"
    )}]
    if memory_context:
        messages.append({"role": "system", "content": memory_context})

    # 加入对话历史（最近 10 轮）
    for msg in history[-20:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    # 3. 调用 DeepSeek
    try:
        response = chat_llm.invoke(messages)
        assistant_msg = response.content
    except Exception as e:
        assistant_msg = f"调用 DeepSeek 出错: {e}"

    # 4. 更新历史
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_msg})

    # 5. 写入记忆
    ingest_turn(user_message, assistant_msg)

    # 6. 场景管理更新（视锥剔除等）
    import re
    words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]{3,}", user_message)
    entities = list(set(w.lower() for w in words if len(w) >= 2))[:5]
    scene.update(user_message, current_entities=entities)

    return history, get_memory_status()


def get_memory_status() -> str:
    """获取记忆引擎实时状态。"""
    s = scene.stats()
    total = registry.count_memories()
    clusters = cluster_mgr.list_clusters()

    lines = [f"📊 总记忆: {total} | 热区: {s['hot']} | 温区: {s['warm']} | 冷区: {s['cold']} | 轮次: {s['turn_count']}"]

    hot = scene.cache.get_hot()
    if hot:
        lines.append(f"\n🔴 热区 ({len(hot)}):")
        for m in hot[:5]:
            lines.append(f"  {m.id}: {m.label[:50]}")

    warm = scene.cache.get_warm()
    if warm:
        lines.append(f"\n🟡 温区 ({len(warm)}):")
        for m in warm[:5]:
            lines.append(f"  {m.id}: {m.label[:50]}")

    if clusters:
        lines.append(f"\n📦 记忆簇 ({len(clusters)}):")
        for c in clusters:
            lines.append(f"  {c.name} ({len(c.memory_ids)} 条)")

    return "\n".join(lines)


# === Gradio UI ===

with gr.Blocks(title="MemoryAtlas + DeepSeek Agent") as app:
    gr.Markdown("# 🧠 MemoryAtlas + DeepSeek 对话 Agent")
    gr.Markdown("带长期记忆的真实对话。每轮自动：检索记忆 → 注入上下文 → 对话 → 写入记忆 → 场景管理。")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话", height=500)
            msg_input = gr.Textbox(
                label="输入消息",
                placeholder="试试: 我在做一个用户认证模块，用 JWT...",
                lines=2,
            )
            with gr.Row():
                send_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清空对话")

        with gr.Column(scale=1):
            memory_status = gr.Textbox(
                label="🗄️ 记忆引擎状态",
                value=get_memory_status(),
                lines=25,
                interactive=False,
            )

    state = gr.State([])

    def send(user_msg, history):
        history, status = chat(user_msg, history)
        return "", history, history, status

    def clear():
        scene.cache.clear()
        scene.initialize_session("chat_session_new")
        return [], [], get_memory_status()

    msg_input.submit(send, [msg_input, state], [msg_input, chatbot, state, memory_status])
    send_btn.click(send, [msg_input, state], [msg_input, chatbot, state, memory_status])
    clear_btn.click(clear, [], [chatbot, state, memory_status])

    gr.Markdown("""
    ### 💡 试试这些对话场景
    1. 先聊 JWT 认证相关话题（几轮），观察记忆积累和自动归簇
    2. 说"换个话题，聊聊数据库优化"，观察热区记忆被降级
    3. 再回到 JWT 话题，看 agent 是否还记得之前的讨论
    """)

if __name__ == "__main__":
    app.launch()
