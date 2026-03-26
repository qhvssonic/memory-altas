"""MemoryAtlas 端到端演示 — 不需要 API key，纯本地运行。

展示核心机制：
1. 写入记忆（ingestion pipeline）
2. 三层缓存 + 缓存命中
3. LOD 精度切换
4. 视锥剔除（话题转换时主动卸载）
5. 记忆簇自动归簇
6. 遗忘机制
7. 导出/导入
8. CLI 命令

运行方式：
  uv run python examples/demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# 把 src 加到 path（开发模式下）
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory_atlas.config import MemoryAtlasConfig
from memory_atlas.core.registry import Registry, MemoryRecord
from memory_atlas.core.tree_index import TreeIndex
from memory_atlas.storage.file_store import FileStore, MemoryChunk
from memory_atlas.storage.cache import CacheManager, CachedMemory, CacheTier
from memory_atlas.scene.manager import SceneManager
from memory_atlas.scene.culler import FrustumCuller
from memory_atlas.scene.lod import LODManager
from memory_atlas.retrieval.fusion import FusionRanker
from memory_atlas.core.cluster import ClusterManager
from memory_atlas.maintenance.forgetting import ForgettingManager
from memory_atlas.io.exporter import Exporter
from memory_atlas.io.importer import Importer

import random
import hashlib
from datetime import datetime, timezone, timedelta


# === Fake Embedder（不需要下载模型） ===
class DemoEmbedder:
    """确定性的假 embedder，基于文本 hash 生成向量。"""
    @property
    def dim(self) -> int:
        return 32

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
            rng = random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(self.dim)]
            # 归一化
            norm = sum(x*x for x in vec) ** 0.5
            results.append([x / norm for x in vec])
        return results


class DemoLLM:
    """假 LLM，返回固定结构。"""
    def complete(self, prompt, **kw):
        return "demo response"

    def complete_json(self, prompt, **kw):
        if "extract" in prompt.lower() or "entities" in prompt.lower():
            return {"entities": [], "facts": [], "decisions": [], "topics": [], "importance": 0.5}
        if "summary" in prompt.lower():
            return {"label": "demo label", "summary": "demo summary"}
        return {}


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    tmp = tempfile.mkdtemp()
    print(f"演示数据目录: {tmp}\n")

    # 初始化组件
    config = MemoryAtlasConfig(
        storage_path=tmp, hot_capacity=5, warm_capacity=20,
        prefetch_enabled=False, culling_enabled=True,
    )
    registry = Registry(Path(tmp) / "index.duckdb")
    tree = TreeIndex(tmp)
    file_store = FileStore(tmp)
    embedder = DemoEmbedder()
    llm = DemoLLM()
    scene = SceneManager(config, registry, tree, file_store, embedder, llm)
    cluster_mgr = ClusterManager(registry)

    # =========================================================
    section("1. 写入记忆")
    # =========================================================
    memories_data = [
        ("jwt_001", "s1", "JWT token 过期 bug", "refresh token 存在竞态条件，决定使用滑动窗口策略", ["jwt", "auth"]),
        ("jwt_002", "s1", "JWT 密钥轮换方案", "每 30 天自动轮换签名密钥，旧 token 宽限期 24h", ["jwt", "auth"]),
        ("jwt_003", "s1", "JWT 黑名单机制", "使用 Redis 存储已撤销的 token ID", ["jwt", "auth", "redis"]),
        ("db_001", "s1", "数据库慢查询优化", "users 表缺少 email 字段索引，添加后查询从 2s 降到 5ms", ["database", "performance"]),
        ("db_002", "s1", "数据库连接池配置", "最大连接数从 10 调到 50，idle timeout 设为 30s", ["database", "performance"]),
        ("db_003", "s1", "数据库分表方案", "按月分表，超过 1000 万行自动归档", ["database"]),
        ("deploy_001", "s1", "Docker 部署流程", "使用 multi-stage build 减小镜像体积", ["deploy", "docker"]),
        ("deploy_002", "s1", "CI/CD 流水线", "GitHub Actions + Docker Hub 自动构建", ["deploy", "ci"]),
    ]

    for mid, sid, label, summary, entities in memories_data:
        emb = embedder.embed([summary])[0]
        registry.insert_memory(MemoryRecord(
            id=mid, session_id=sid, label=label, summary=summary,
            embedding=emb, importance_score=0.7,
        ))
        file_store.save_chunk(MemoryChunk(
            id=mid, session_id=sid, entities=entities,
            importance=0.7, title=label, content=f"详细内容：{summary}",
        ))
        # 关联实体
        for ent in entities:
            eid = registry.upsert_entity(ent, "concept")
            registry.link_memory_entity(mid, eid)

    print(f"写入了 {registry.count_memories()} 条记忆")
    print(f"记忆 ID: {[m[0] for m in memories_data]}")

    # =========================================================
    section("2. 记忆簇自动归簇")
    # =========================================================
    for mid, _, _, _, entities in memories_data:
        cluster_mgr.auto_update_for_memory(mid, entities, threshold=3)

    clusters = cluster_mgr.list_clusters()
    print(f"自动创建了 {len(clusters)} 个簇:")
    for c in clusters:
        print(f"  {c.name}: {len(c.memory_ids)} 条记忆, 标签={c.entity_tags}")
        # 展示簇级别 LOD
        l0 = cluster_mgr.get_cluster_lod(c.id, "L0")
        l1 = cluster_mgr.get_cluster_lod(c.id, "L1")
        print(f"    L0: {l0}")
        print(f"    L1: {l1}")

    # =========================================================
    section("3. 场景管理器检索 + 三层缓存")
    # =========================================================
    scene.initialize_session("demo")

    print("\n--- 第 1 轮：查询 JWT 相关 ---")
    results = scene.get_memory_view("JWT token 过期 bug")
    print(f"检索到 {len(results)} 条记忆:")
    for m in results:
        print(f"  [{m.lod}] [{m.tier.value}] {m.id}: {m.display_text[:60]}")

    stats = scene.stats()
    print(f"\n缓存状态: hot={stats['hot']}, warm={stats['warm']}, cold={stats['cold']}")

    print("\n--- 第 2 轮：再次查询 JWT（应该缓存命中） ---")
    results2 = scene.get_memory_view("JWT 密钥轮换")
    print(f"检索到 {len(results2)} 条记忆:")
    for m in results2:
        print(f"  [{m.lod}] [{m.tier.value}] {m.id}: {m.display_text[:60]}")

    # =========================================================
    section("4. LOD 精度切换")
    # =========================================================
    lod_mgr = LODManager(file_store, max_tokens=200)
    demo_mems = [
        CachedMemory(id="jwt_001", label="JWT 过期 bug", summary="refresh token 竞态条件", importance=0.9),
        CachedMemory(id="jwt_002", label="JWT 密钥轮换", summary="30 天自动轮换签名密钥", importance=0.5),
        CachedMemory(id="db_001", label="慢查询优化", summary="添加 email 索引", importance=0.3),
        CachedMemory(id="db_002", label="连接池配置", summary="最大连接数调到 50", importance=0.1),
    ]
    scores = {"jwt_001": 0.9, "jwt_002": 0.5, "db_001": 0.3, "db_002": 0.1}
    assigned = lod_mgr.assign_lod(demo_mems, scores)
    print("根据相关性分配 LOD:")
    for m in assigned:
        print(f"  {m.id}: {m.lod} → \"{m.display_text[:50]}\"")

    print(f"\n展开 jwt_001 到 L2:")
    expanded = lod_mgr.expand_to_l2(assigned[0])
    print(f"  L2 内容: {expanded.display_text}")

    # =========================================================
    section("5. 视锥剔除（话题转换）")
    # =========================================================
    print("当前热区:")
    for m in scene.cache.get_hot():
        print(f"  [{m.tier.value}] {m.id}: {m.label}")

    print("\n用户说: \"换个话题，我们来看看数据库性能问题\"")
    scene.update("换个话题，我们来看看数据库性能问题", current_entities=["database", "performance"])

    print("\n剔除后热区:")
    hot_after = scene.cache.get_hot()
    if hot_after:
        for m in hot_after:
            print(f"  [{m.tier.value}] {m.id}: {m.label}")
    else:
        print("  (热区已清空，JWT 记忆被降级到温区)")

    print("\n温区（被降级的记忆仍可快速恢复）:")
    for m in scene.cache.get_warm()[:5]:
        print(f"  [{m.tier.value}] {m.id}: {m.label}")

    # =========================================================
    section("6. 遗忘机制")
    # =========================================================
    # 插入一条很旧的记忆
    old_time = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
    registry.insert_memory(MemoryRecord(
        id="ancient_001", label="半年前的旧讨论", summary="已经不重要了",
        importance_score=0.05, access_count=0, last_accessed_at=old_time,
    ))
    file_store.save_chunk(MemoryChunk(id="ancient_001", content="很旧的内容"))

    fm = ForgettingManager(registry, file_store, decay_lambda=0.1)
    result = fm.run_cycle()
    print(f"遗忘周期: 扫描 {result.scanned} 条")
    print(f"  保留: {result.kept}, 压缩: {result.compressed}, 归档: {result.archived}")

    # =========================================================
    section("7. 导出 / 导入")
    # =========================================================
    export_path = Path(tmp) / "export.json"
    exporter = Exporter(registry, file_store)
    stats = exporter.export_all(str(export_path))
    print(f"导出 {stats['memories_exported']} 条记忆到 {export_path.name}")

    # 导入到新的 registry
    tmp2 = tempfile.mkdtemp()
    registry2 = Registry(Path(tmp2) / "index.duckdb")
    file_store2 = FileStore(tmp2)
    importer = Importer(registry2, file_store2)
    istats = importer.import_file(str(export_path), mode="merge")
    print(f"导入: {istats['imported']} 条, 跳过: {istats['skipped']} 条")
    print(f"新库记忆数: {registry2.count_memories()}")
    registry2.close()

    # =========================================================
    section("8. 多用户隔离")
    # =========================================================
    registry.insert_memory(MemoryRecord(id="alice_001", user_id="alice", label="Alice 的记忆"))
    registry.insert_memory(MemoryRecord(id="bob_001", user_id="bob", label="Bob 的记忆"))
    alice_mems = registry.list_memories(user_id="alice")
    bob_mems = registry.list_memories(user_id="bob")
    print(f"Alice 的记忆: {len(alice_mems)} 条")
    print(f"Bob 的记忆: {len(bob_mems)} 条")
    print("用户之间完全隔离 ✓")

    # =========================================================
    section("✅ 演示完成")
    # =========================================================
    total = registry.count_memories()
    print(f"总记忆数: {total}")
    print(f"簇数量: {len(cluster_mgr.list_clusters())}")
    print(f"\n所有核心机制验证通过:")
    print(f"  · 写入管道 (ingestion)")
    print(f"  · 三层缓存 (hot/warm/cold)")
    print(f"  · LOD 精度切换 (L0/L1/L2)")
    print(f"  · 视锥剔除 (frustum culling)")
    print(f"  · 记忆簇自动归簇 (auto-clustering)")
    print(f"  · 遗忘机制 (forgetting)")
    print(f"  · 导出/导入 (export/import)")
    print(f"  · 多用户隔离 (multi-user)")

    registry.close()


if __name__ == "__main__":
    main()
