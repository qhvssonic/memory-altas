"""MemoryAtlas Gradio 交互界面 — 可视化验证记忆引擎。

运行: uv run python examples/gradio_app.py
然后浏览器打开 http://localhost:7860
"""

from __future__ import annotations

import sys
import tempfile
import hashlib
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gradio as gr

from memory_atlas.config import MemoryAtlasConfig
from memory_atlas.core.registry import Registry, MemoryRecord
from memory_atlas.core.tree_index import TreeIndex
from memory_atlas.storage.file_store import FileStore, MemoryChunk
from memory_atlas.storage.cache import CachedMemory, CacheTier
from memory_atlas.scene.manager import SceneManager
from memory_atlas.scene.lod import LODManager
from memory_atlas.core.cluster import ClusterManager
from memory_atlas.maintenance.forgetting import ForgettingManager


# === Fake Embedder ===
class DemoEmbedder:
    @property
    def dim(self) -> int:
        return 32

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
            rng = random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(self.dim)]
            norm = sum(x * x for x in vec) ** 0.5
            results.append([x / norm for x in vec])
        return results


class DemoLLM:
    def complete(self, prompt, **kw):
        return "demo"

    def complete_json(self, prompt, **kw):
        return {}


# === Global State ===
TMP_DIR = tempfile.mkdtemp()
config = MemoryAtlasConfig(
    storage_path=TMP_DIR, hot_capacity=10, warm_capacity=50,
    prefetch_enabled=False, culling_enabled=True,
)
registry = Registry(Path(TMP_DIR) / "index.duckdb")
tree = TreeIndex(TMP_DIR)
file_store = FileStore(TMP_DIR)
embedder = DemoEmbedder()
llm = DemoLLM()
scene = SceneManager(config, registry, tree, file_store, embedder, llm)
cluster_mgr = ClusterManager(registry)
_counter = [0]

scene.initialize_session("gradio_session")


# === Functions ===

def ingest_memory(content: str, entities_str: str):
    """写入一条记忆。"""
    if not content.strip():
        return "请输入内容", get_stats(), get_cache_view(), get_cluster_view()

    _counter[0] += 1
    mid = f"m{_counter[0]:04d}"
    entities = [e.strip() for e in entities_str.split(",") if e.strip()]
    summary = content[:200]
    emb = embedder.embed([summary])[0]

    registry.insert_memory(MemoryRecord(
        id=mid, session_id="gradio", label=content[:80],
        summary=summary, embedding=emb, importance_score=0.7,
    ))
    file_store.save_chunk(MemoryChunk(
        id=mid, session_id="gradio", entities=entities,
        importance=0.7, title=content[:80], content=content,
    ))
    for ent in entities:
        eid = registry.upsert_entity(ent, "concept")
        registry.link_memory_entity(mid, eid)

    # Auto-cluster
    cluster_mgr.auto_update_for_memory(mid, entities, threshold=3)

    msg = f"✅ 写入成功: {mid} (实体: {entities})"
    return msg, get_stats(), get_cache_view(), get_cluster_view()


def search_memory(query: str):
    """检索记忆。"""
    if not query.strip():
        return "请输入查询", get_stats(), get_cache_view()

    results = scene.get_memory_view(query)
    if not results:
        return "未找到相关记忆", get_stats(), get_cache_view()

    lines = []
    for m in results:
        lines.append(f"[{m.lod}] [{m.tier.value}] {m.id}: {m.display_text[:100]}")
    return "\n".join(lines), get_stats(), get_cache_view()


def do_topic_switch(message: str, entities_str: str):
    """模拟话题转换，触发视锥剔除。"""
    entities = [e.strip() for e in entities_str.split(",") if e.strip()]
    stats = scene.update(message, current_entities=entities)
    msg = f"🔄 话题转换: 剔除 {stats['culled']} 条, 预加载 {stats['prefetched']} 条"
    return msg, get_stats(), get_cache_view()


def do_forget():
    """运行遗忘周期。"""
    fm = ForgettingManager(registry, file_store, decay_lambda=0.1)
    result = fm.run_cycle()
    msg = f"🧹 遗忘: 扫描 {result.scanned}, 保留 {result.kept}, 压缩 {result.compressed}, 归档 {result.archived}"
    return msg, get_stats(), get_cache_view(), get_cluster_view()


def get_stats() -> str:
    """获取引擎统计。"""
    s = scene.stats()
    total = registry.count_memories()
    clusters = len(cluster_mgr.list_clusters())
    return (
        f"总记忆: {total} | 热区: {s['hot']} | 温区: {s['warm']} | "
        f"冷区: {s['cold']} | 簇: {clusters} | 轮次: {s['turn_count']}"
    )


def get_cache_view() -> str:
    """获取缓存状态。"""
    lines = ["=== 热区 (Hot) ==="]
    hot = scene.cache.get_hot()
    if hot:
        for m in hot:
            lines.append(f"  🔴 {m.id}: {m.label[:60]}")
    else:
        lines.append("  (空)")

    lines.append("\n=== 温区 (Warm) ===")
    warm = scene.cache.get_warm()
    if warm:
        for m in warm[:10]:
            lines.append(f"  🟡 {m.id}: {m.label[:60]}")
        if len(warm) > 10:
            lines.append(f"  ... 还有 {len(warm) - 10} 条")
    else:
        lines.append("  (空)")

    return "\n".join(lines)


def get_cluster_view() -> str:
    """获取记忆簇状态。"""
    clusters = cluster_mgr.list_clusters()
    if not clusters:
        return "暂无记忆簇"
    lines = []
    for c in clusters:
        l0 = cluster_mgr.get_cluster_lod(c.id, "L0")
        lines.append(f"📦 {l0} | 标签: {c.entity_tags}")
    return "\n".join(lines)


def load_sample_data():
    """加载示例数据。"""
    samples = [
        ("JWT token 过期 bug: refresh token 存在竞态条件", "jwt,auth"),
        ("JWT 密钥轮换方案: 每 30 天自动轮换签名密钥", "jwt,auth"),
        ("JWT 黑名单机制: 使用 Redis 存储已撤销的 token", "jwt,auth,redis"),
        ("数据库慢查询: users 表缺少 email 索引", "database,performance"),
        ("数据库连接池: 最大连接数从 10 调到 50", "database,performance"),
        ("数据库分表: 按月分表，超过 1000 万行归档", "database"),
        ("Docker 部署: multi-stage build 减小镜像", "deploy,docker"),
        ("CI/CD 流水线: GitHub Actions 自动构建", "deploy,ci"),
    ]
    msgs = []
    for content, entities in samples:
        msg, _, _, _ = ingest_memory(content, entities)
        msgs.append(msg)
    return "\n".join(msgs), get_stats(), get_cache_view(), get_cluster_view()


# === Gradio UI ===

with gr.Blocks(title="MemoryAtlas 记忆引擎", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧠 MemoryAtlas — 游戏引擎式记忆管理")
    gr.Markdown("预测性预加载 · 视锥剔除 · LOD 精度切换 · 三层缓存 · 记忆簇")

    stats_display = gr.Textbox(label="📊 引擎状态", value=get_stats(), interactive=False)

    with gr.Row():
        cache_display = gr.Textbox(label="🗄️ 缓存状态", value=get_cache_view(), lines=12, interactive=False)
        cluster_display = gr.Textbox(label="📦 记忆簇", value=get_cluster_view(), lines=12, interactive=False)

    with gr.Tab("写入记忆"):
        with gr.Row():
            ingest_content = gr.Textbox(label="记忆内容", placeholder="例: JWT refresh token 存在竞态条件")
            ingest_entities = gr.Textbox(label="实体标签 (逗号分隔)", placeholder="例: jwt,auth")
        ingest_btn = gr.Button("写入", variant="primary")
        ingest_result = gr.Textbox(label="结果", interactive=False)
        ingest_btn.click(
            ingest_memory, [ingest_content, ingest_entities],
            [ingest_result, stats_display, cache_display, cluster_display],
        )

    with gr.Tab("检索记忆"):
        search_input = gr.Textbox(label="查询", placeholder="例: token 过期问题")
        search_btn = gr.Button("检索", variant="primary")
        search_result = gr.Textbox(label="检索结果", lines=8, interactive=False)
        search_btn.click(
            search_memory, [search_input],
            [search_result, stats_display, cache_display],
        )

    with gr.Tab("话题转换 (视锥剔除)"):
        gr.Markdown("模拟话题转换，观察热区记忆被降级到温区")
        switch_msg = gr.Textbox(label="转换消息", placeholder="例: 换个话题，我们来看看数据库性能")
        switch_entities = gr.Textbox(label="新话题实体", placeholder="例: database,performance")
        switch_btn = gr.Button("触发话题转换", variant="secondary")
        switch_result = gr.Textbox(label="结果", interactive=False)
        switch_btn.click(
            do_topic_switch, [switch_msg, switch_entities],
            [switch_result, stats_display, cache_display],
        )

    with gr.Tab("遗忘"):
        gr.Markdown("运行遗忘周期，压缩/归档低活跃记忆")
        forget_btn = gr.Button("运行遗忘", variant="secondary")
        forget_result = gr.Textbox(label="结果", interactive=False)
        forget_btn.click(
            do_forget, [],
            [forget_result, stats_display, cache_display, cluster_display],
        )

    with gr.Tab("加载示例数据"):
        gr.Markdown("一键加载 8 条示例记忆（JWT/数据库/部署），观察自动归簇")
        sample_btn = gr.Button("加载示例数据", variant="primary")
        sample_result = gr.Textbox(label="结果", lines=10, interactive=False)
        sample_btn.click(
            load_sample_data, [],
            [sample_result, stats_display, cache_display, cluster_display],
        )


if __name__ == "__main__":
    app.launch()
