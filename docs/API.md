# MemoryAtlas API 文档

## 快速开始

```python
from memory_atlas import MemoryEngine

engine = MemoryEngine(storage_path="./memory", embedding_model="local")
engine.ingest("JWT refresh token 存在竞态条件，决定使用滑动窗口策略。")
results = engine.retrieve("token 过期问题")
engine.close()
```

## 核心类

### MemoryEngine

框架无关的核心引擎，封装了完整的写入/检索/场景管理流程。

```python
from memory_atlas import MemoryEngine

engine = MemoryEngine(
    storage_path="./my_memory",   # 数据存储目录
    embedding_model="local",      # "local" | "openai" | "cohere"
    llm_model="openai/gpt-4o-mini",
    max_memory_tokens=2000,       # 记忆上下文 token 预算
    hot_capacity=20,              # 热区容量
    warm_capacity=100,            # 温区容量
    prefetch_enabled=True,        # 预测性预加载
    culling_enabled=True,         # 视锥剔除
    ingest_strategy="rule_based", # "rule_based" | "llm"
    decay_lambda=0.1,             # 遗忘衰减系数
)
```

#### 方法

| 方法 | 说明 | 返回值 |
|---|---|---|
| `ingest(content, session_id="")` | 完整写入管道：分段→提取→摘要→向量化→存储 | `list[str]` 记忆 ID 列表 |
| `maybe_ingest(content, session_id="")` | 条件写入（规则/LLM 判断是否值得记忆） | `list[str] \| None` |
| `bulk_ingest(conversations, session_id="")` | 批量写入多段内容 | `list[str]` |
| `retrieve(query, top_k=10)` | 场景管理器检索：缓存优先→冷区双路搜索→LOD | `list[CachedMemory]` |
| `expand(memory_id)` | 展开到 L2 完整内容 | `CachedMemory \| None` |
| `format_memories(memories)` | 格式化为可注入 prompt 的文本 | `str` |
| `forget(limit=500)` | 运行遗忘周期：压缩/归档低活跃记忆 | `ForgetResult` |
| `stats()` | 引擎统计信息 | `dict` |
| `close()` | 持久化并释放资源 | `None` |

---

### MemoryAtlasMiddleware

LangChain 1.0 AgentMiddleware 集成，一行接入。

```python
from memory_atlas.langchain import MemoryAtlasMiddleware

memory = MemoryAtlasMiddleware(
    storage_path="./memory",
    embedding_model="local",
    max_memory_tokens=2000,
)
# 传入 agent 的 middleware 列表即可
```

#### 钩子

| 钩子 | 触发时机 | 行为 |
|---|---|---|
| `before_agent(state, runtime)` | Agent 启动 | 初始化会话缓存 |
| `before_model(state, runtime)` | LLM 调用前 | 检索记忆注入 messages |
| `after_model(state, runtime)` | LLM 调用后 | 评估写入 + 场景管理更新 |
| `after_agent(state, runtime)` | Agent 结束 | 持久化 + 学习话题转换 |

---

## 场景管理器

### SceneManager

核心差异化组件，编排预加载/剔除/LOD。

```python
from memory_atlas.scene.manager import SceneManager

scene = SceneManager(config, registry, tree, file_store, embedder, llm)
scene.initialize_session("session_001")
memories = scene.get_memory_view("当前查询")
scene.update("最近消息", current_entities=["auth", "jwt"])
scene.persist()
```

| 方法 | 说明 |
|---|---|
| `initialize_session(session_id)` | 初始化会话状态 |
| `get_memory_view(query)` | 获取当前最优记忆视野（缓存优先） |
| `update(recent_message, current_entities)` | 场景更新：预加载 + 剔除 |
| `expand_memory(memory_id)` | 展开到 L2 |
| `format_context(memories)` | 格式化上下文字符串 |
| `stats()` | 统计信息 |
| `persist()` | 持久化树索引 |

### Prefetcher

预测下一轮可能需要的记忆，预加载到温区。

三种策略：
- 话题转换历史（`topic_transitions` 表）
- 实体扩展（关联实体的记忆）
- LLM 预测（预测下一步话题）

### FrustumCuller

检测话题转换，主动降级不相关记忆。

检测信号：
- 显式："换个话题"、"let's move on" 等
- 隐式：实体重叠度低于阈值
- 漂移：连续 N 轮未引用

### LODManager

三级精度动态切换：
- L0（标签）：~20 tokens
- L1（摘要）：~80 tokens
- L2（完整）：~500+ tokens

---

## 存储层

### CacheManager

三层缓存：Hot（内存 dict）→ Warm（LRU）→ Cold（磁盘）。

```python
from memory_atlas.storage.cache import CacheManager, CachedMemory

cache = CacheManager(hot_capacity=20, warm_capacity=100)
cache.promote_to_hot(memory)
cache.demote_to_warm(memory_id)
cache.get(memory_id)  # 查找 hot/warm
```

### Registry

DuckDB 元数据索引，6 张表。

```python
from memory_atlas.core.registry import Registry, MemoryRecord

reg = Registry("./data/index.duckdb")
reg.insert_memory(MemoryRecord(id="m1", label="...", embedding=[...]))
results = reg.vector_search(query_embedding, top_k=10)
reg.upsert_entity("jwt", "concept")
reg.record_transition("auth", "database", "session_1")
```

### FileStore

Markdown 文件存储（L2 原始内容）。

```python
from memory_atlas.storage.file_store import FileStore, MemoryChunk

store = FileStore("./data")
store.save_chunk(MemoryChunk(id="c001", content="..."))
chunk = store.load_chunk("c001")
```

### TreeIndex

JSON 树状语义索引。

```python
from memory_atlas.core.tree_index import TreeIndex, TreeNode

tree = TreeIndex("./data")
tree.add_child("root", TreeNode(id="auth", label="Authentication"))
tree.add_memory_to_node("auth", "m001")
outline = tree.get_outline(max_depth=2)
```

---

## 写入管道

### Chunker

语义分段，支持 `turn`（对话轮次）、`paragraph`（段落）、`fixed`（固定长度）。

### Extractor

LLM 信息提取（实体/事实/决策/话题/重要性），带规则回退。

### Summarizer

多级摘要生成：L0 标签 + L1 摘要，带规则回退。

---

## 检索引擎

### VectorSearch

DuckDB `list_cosine_similarity` 向量相似度搜索。

### TreeSearch

LLM 导航树状索引，深度优先推理检索。

### FusionRanker

双路融合排序：向量权重 + 树权重 + 双路命中 20% boost。

---

## 维护

### ForgettingManager

基于活跃度衰减的遗忘机制。

```python
from memory_atlas.maintenance.forgetting import ForgettingManager

fm = ForgettingManager(registry, file_store, decay_lambda=0.1)
result = fm.run_cycle(limit=500)
# result.compressed — L2 文件删除，保留 L0+L1
# result.archived — 仅保留 L0 标签
# result.kept — 活跃度足够，保留
```

活跃度公式：`activity = importance × e^(-λ × days) × log(access_count + 2)`

---

## 配置

### MemoryAtlasConfig

所有配置项及默认值：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `storage_path` | `"./memory_atlas_data"` | 数据存储目录 |
| `embedding_model` | `"local"` | 嵌入模型：local/openai/cohere |
| `embedding_dim` | `384` | 向量维度 |
| `llm_model` | `"openai/gpt-4o-mini"` | LLM 模型 |
| `max_memory_tokens` | `2000` | 记忆上下文 token 预算 |
| `hot_capacity` | `20` | 热区容量 |
| `warm_capacity` | `100` | 温区容量 |
| `prefetch_enabled` | `True` | 启用预测性预加载 |
| `prefetch_top_k` | `10` | 预加载数量 |
| `culling_enabled` | `True` | 启用视锥剔除 |
| `culling_overlap_threshold` | `0.3` | 实体重叠阈值 |
| `lod_default` | `"L1"` | 默认 LOD 级别 |
| `ingest_strategy` | `"rule_based"` | 写入策略：rule_based/llm |
| `ingest_min_length` | `50` | 规则模式最小长度 |
| `retrieval_top_k` | `10` | 检索返回数量 |
| `vector_weight` | `0.6` | 向量路权重 |
| `tree_weight` | `0.4` | 树状路权重 |
| `decay_lambda` | `0.1` | 遗忘衰减系数 |

---

## 数据模型

### CachedMemory

缓存中的记忆条目。

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `str` | 记忆 ID |
| `label` | `str` | L0 标签 |
| `summary` | `str` | L1 摘要 |
| `content` | `str \| None` | L2 完整内容（按需加载） |
| `embedding` | `list[float]` | 向量 |
| `importance` | `float` | 重要性 0-1 |
| `tier` | `CacheTier` | hot/warm/cold |
| `lod` | `str` | 当前精度 L0/L1/L2 |
| `entities` | `list[str]` | 关联实体 |
| `display_text` | `str` | 当前 LOD 下的展示文本（属性） |
| `token_estimate()` | `int` | 当前 LOD 的 token 估算 |

### MemoryRecord

DuckDB 中的记忆行记录，字段对应 `memories` 表。

### MemoryChunk

Markdown 文件的内存表示，含 frontmatter 元数据。
