# MemoryAtlas — 设计方案

> 第一个用游戏引擎资源管理思想设计的智能体记忆系统

## 项目定位

MemoryAtlas 是一个面向 AI 应用开发者的 Python SDK，为 LangChain 1.0 agent 提供透明的记忆管理能力。

现有记忆系统（Mem0、Zep、Letta 等）都是"被动检索"模式——query 来了，搜索，返回结果。
MemoryAtlas 的核心差异是引入游戏引擎的"主动场景管理"——像游戏引擎管理纹理和模型一样，
主动预测、预加载、分级、剔除记忆，让 agent 在任意时刻都拥有最优的记忆视野。

## 竞品对比与差异化定位

| 特性 | Mem0 | Letta | Zep | Hindsight | MemoryAtlas |
|---|---|---|---|---|---|
| 检索模式 | 被动（query→search） | 被动（agent 系统调用） | 被动（图遍历） | 被动（四路并行） | **主动场景管理** |
| 预加载 | ✗ | ✗ | ✗ | ✗ | **✓ 预测性预加载** |
| 视锥剔除 | ✗ | ✗ | ✗ | ✗ | **✓ 主动排除不相关记忆** |
| 多级精度（LOD） | ✗ | 部分（core/archival） | ✗ | ✗ | **✓ L0/L1/L2 三级** |
| 缓存分层 | ✗ | ✗ | ✗ | ✗ | **✓ 热/温/冷三层** |
| 外部依赖 | 向量数据库 | 完整运行时 | Neo4j | Docker 全栈 | **零依赖，pip install** |
| 存储 | 云服务/自建 | 服务端 | 服务端 | 服务端 | **本地文件，Git 友好** |
| LangChain 1.0 原生 | ✗ | ✗ | ✗ | ✗ | **✓ AgentMiddleware** |

## 核心创新：游戏引擎式记忆管理

### 为什么现有方案不够？

所有现有记忆系统的工作模式：

```
用户消息 → 生成 query → 搜索记忆库 → 返回 top-k → 注入 prompt
```

问题在于：
1. **每次都是冷启动**——不管上一轮聊了什么，每轮都从零开始搜索
2. **无法预判**——不会根据对话走向提前准备记忆
3. **无法主动遗忘**——用户明确转换话题后，旧记忆仍然占据上下文
4. **精度单一**——要么返回完整内容（浪费 token），要么返回摘要（可能丢失细节）

### MemoryAtlas 的解法：场景管理引擎

游戏引擎面对同样的问题——世界很大，但玩家视野有限。引擎的解法：

```
┌─────────────────────────────────────────────────────────────┐
│                    Scene Manager（场景管理器）                 │
│                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │  Prefetcher   │  │  LOD Manager  │  │  Frustum Culler│  │
│  │  预加载器      │  │  精度管理器    │  │  视锥剔除器     │  │
│  │               │  │               │  │                │  │
│  │  根据对话方向  │  │  远处用 L0    │  │  话题转换时     │  │
│  │  预测下一步    │  │  近处用 L1    │  │  主动卸载       │  │
│  │  需要的记忆    │  │  聚焦用 L2    │  │  不相关记忆     │  │
│  └───────┬───────┘  └───────┬───────┘  └───────┬────────┘  │
│          │                  │                   │           │
│          ▼                  ▼                   ▼           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Memory View（记忆视野）                  │    │
│  │                                                     │    │
│  │  agent 当前上下文中的记忆，始终是：                     │    │
│  │  · 最相关的                                         │    │
│  │  · 最合适精度的                                      │    │
│  │  · 最省 token 的                                    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 创新点 1：预测性预加载（Prefetching）

现有方案：用户说了什么 → 搜索相关记忆（被动）

MemoryAtlas：用户说了什么 → 搜索相关记忆 + **预测下一步可能聊什么 → 提前加载到温区**（主动）

```
用户："我在做用户认证模块"

  被动检索（所有竞品）：
    → 搜索"用户认证"相关记忆，返回结果，结束

  MemoryAtlas 额外做的：
    → 预测接下来可能聊：JWT、session、OAuth、密码加密、部署
    → 从冷区预加载这些话题的记忆到温区
    → 下一轮用户真的聊到 JWT 时，记忆已经在温区，零延迟
```

预测策略（可叠加）：
- **话题关联**：历史上 A 话题之后经常聊 B，预加载 B
- **实体扩展**：提到"auth 模块"，预加载所有 auth 相关记忆
- **时间局部性**：最近的记忆更可能被再次用到
- **会话模式**：学习用户的对话习惯模式

### 创新点 2：视锥剔除（Frustum Culling）

游戏里玩家背后的东西不渲染。MemoryAtlas 主动检测话题转换，卸载不相关记忆。

```
用户："好，认证的事先放一放，我们来看看数据库性能问题"

  现有方案：
    → 新搜索"数据库性能"，但上一轮的 auth 记忆可能还在上下文里占 token

  MemoryAtlas：
    → 检测到明确的话题转换信号
    → 主动将 auth 相关记忆从热区降级到温区
    → 腾出 token 空间给数据库性能相关记忆
    → 如果用户又回到 auth 话题，从温区快速恢复（不用重新搜索冷区）
```

剔除信号检测：
- **显式信号**：用户说"换个话题"、"先不管这个"
- **隐式信号**：连续 N 轮没有引用某话题的记忆
- **实体漂移**：当前对话涉及的实体与热区记忆的实体重叠度低于阈值

### 创新点 3：三级精度动态切换（LOD）

不是所有记忆都需要同样的细节。MemoryAtlas 根据相关性动态选择精度：

```
记忆精度层级：

L0（标签）："2024-03 讨论了 auth 模块的 JWT token 过期 bug"
  → ~20 tokens，用于索引和粗筛

L1（摘要）："refresh token 存在竞态条件，决定使用滑动窗口策略。涉及 auth/token.ts"
  → ~80 tokens，大多数场景够用

L2（完整）：原始对话记录，包含所有细节和上下文
  → ~500+ tokens，只在需要精确细节时加载
```

动态切换逻辑：
- 检索到 10 条相关记忆 → 全部以 L0 展示给 agent
- Agent 判断其中 3 条高度相关 → 自动提升到 L1
- Agent 需要某条的具体细节 → 按需展开到 L2
- **总 token 消耗始终可控**，不会因为记忆多就爆上下文

### 三层缓存架构

```
┌─────────────────────────────────────────────────────┐
│  热区（Hot）                                         │
│  当前上下文中正在使用的记忆                             │
│  数据结构：内存字典，O(1) 访问                         │
│  容量：受 max_memory_tokens 限制                      │
├─────────────────────────────────────────────────────┤
│  温区（Warm）                                        │
│  预加载的候选记忆 + 最近降级的记忆                      │
│  数据结构：内存 LRU 缓存                              │
│  容量：热区的 3-5 倍                                  │
│  特点：提升到热区零成本，不需要磁盘 IO                  │
├─────────────────────────────────────────────────────┤
│  冷区（Cold）                                        │
│  全量存储（DuckDB + 文件系统）                         │
│  容量：无限                                          │
│  检索：向量搜索 + 树状推理，需要磁盘 IO                │
└─────────────────────────────────────────────────────┘

状态流转：
  冷 → 温：预加载器根据话题预测提升
  温 → 热：before_model 检索命中时提升
  热 → 温：视锥剔除器检测到话题转换时降级
  温 → 冷：LRU 淘汰或长时间未访问
```

## 游戏引擎概念映射（完整）

| 游戏引擎概念 | 记忆系统对应 | 竞品是否有 |
|---|---|---|
| 资产注册表（Asset Registry） | DuckDB 元数据索引 | Mem0 有类似（向量DB） |
| LOD（Level of Detail） | L0/L1/L2 三级记忆精度 | Letta 有部分 |
| 流式预加载（Streaming/Prefetch） | 预测性预加载到温区 | **无竞品** |
| 空间分区（Spatial Partitioning） | 树状语义索引 | Zep 有图结构 |
| 视锥剔除（Frustum Culling） | 主动排除不相关记忆 | **无竞品** |
| 资产打包（Asset Bundling） | 记忆簇（v0.2） | 无 |
| Chunk Loading/Unloading | 热/温/冷三层缓存 | **无竞品** |
| 场景管理器（Scene Manager） | 记忆视野管理 | **无竞品** |

## 技术选型

| 组件 | 选型 | 理由 |
|---|---|---|
| 语言 | Python 3.10+ | LLM 生态最成熟，社区贡献门槛低 |
| 包管理 | uv | 速度快，现代 Python 趋势 |
| 元数据索引 | DuckDB | 嵌入式、单文件、列式存储、支持向量运算，零外部依赖 |
| 记忆本体存储 | Markdown 文件 | 人类可读、Git 友好 |
| 树状索引 | JSON 文件 | 描述记忆层级结构，轻量 |
| LLM 接口 | LiteLLM | 统一接口，支持 OpenAI/Anthropic/Ollama |
| Embedding（本地） | sentence-transformers (all-MiniLM-L6-v2) | 离线可用、体积小 |
| Embedding（云端） | OpenAI / Cohere（可选） | 质量更高，用户自选 |
| License | MIT | 最宽松 |

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                      MemoryAtlas SDK                         │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │           Scene Manager（核心差异化）                    │  │
│  │                                                        │  │
│  │  Prefetcher ──── LOD Manager ──── Frustum Culler       │  │
│  │  预测性预加载      精度动态切换      主动剔除不相关       │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│  ┌────────────┐  ┌────────┴───────┐  ┌─────────────────┐    │
│  │ Ingestion  │  │   Retrieval    │  │  Cache Manager  │    │
│  │ Pipeline   │  │   Engine       │  │                 │    │
│  │            │  │                │  │  Hot  (内存)     │    │
│  │ · Chunker  │  │ · Vector (DuckDB)│ │  Warm (LRU)    │    │
│  │ · Extractor│  │ · Tree Nav     │  │  Cold (磁盘)    │    │
│  │ · Summary  │  │ · Fusion       │  │                 │    │
│  └─────┬──────┘  └────────┬───────┘  └────────┬────────┘    │
│        │                  │                    │             │
│        ▼                  ▼                    ▼             │
│  ┌───────────────────────────────────────────────────────┐   │
│  │                   Storage Layer                        │   │
│  │  DuckDB (index.duckdb)  │  Markdown (chunks/*.md)     │   │
│  │  元数据 + 向量           │  记忆本体（L2）              │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  LangChain 1.0 Integration                            │   │
│  │  MemoryAtlasMiddleware (AgentMiddleware)               │   │
│  │  before_agent → before_model → after_model → after_agent│  │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## 数据模型

### DuckDB 表结构

```sql
-- 记忆主表
CREATE TABLE memories (
    id VARCHAR PRIMARY KEY,
    session_id VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    label VARCHAR,              -- L0 标签
    summary TEXT,               -- L1 摘要
    file_path VARCHAR,          -- L2 原始文件路径
    embedding FLOAT[],          -- 向量
    importance_score FLOAT,     -- 重要性评分 0-1
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP,
    parent_node VARCHAR,        -- 树状索引中的父节点
    cache_tier VARCHAR DEFAULT 'cold',  -- hot / warm / cold
    metadata JSON               -- 扩展字段
);

-- 实体表
CREATE TABLE entities (
    id VARCHAR PRIMARY KEY,
    name VARCHAR,
    type VARCHAR,               -- file / function / concept / person / project
    first_seen TIMESTAMP,
    last_seen TIMESTAMP
);

-- 记忆-实体关联
CREATE TABLE memory_entities (
    memory_id VARCHAR,
    entity_id VARCHAR,
    relation VARCHAR,           -- mentions / about / decides / creates
    PRIMARY KEY (memory_id, entity_id)
);

-- 记忆关联
CREATE TABLE memory_links (
    source_id VARCHAR,
    target_id VARCHAR,
    link_type VARCHAR,          -- follows / contradicts / refines / replaces
    PRIMARY KEY (source_id, target_id)
);

-- 树状索引节点
CREATE TABLE tree_nodes (
    id VARCHAR PRIMARY KEY,
    parent_id VARCHAR,
    label VARCHAR,
    summary TEXT,
    node_type VARCHAR,          -- root / category / topic / memory
    depth INTEGER,
    children_count INTEGER DEFAULT 0
);

-- 话题转换日志（用于训练预加载模型）
CREATE TABLE topic_transitions (
    id VARCHAR PRIMARY KEY,
    from_topic VARCHAR,
    to_topic VARCHAR,
    session_id VARCHAR,
    timestamp TIMESTAMP,
    transition_count INTEGER DEFAULT 1
);
```

### 记忆文件格式 (chunks/*.md)

```markdown
---
id: c001
session_id: s_20240315
created_at: 2024-03-15T10:30:00Z
entities: [auth, jwt, refresh-token]
importance: 0.85
---

# 讨论 auth 模块的 JWT token 过期 bug

## 上下文
用户在开发基于 Next.js + Prisma 的认证系统...

## 关键内容
（完整对话/内容记录）

## 提取的事实
- refresh token 逻辑存在竞态条件
- 决定使用滑动窗口策略解决
```

## 核心流程

### 写入流程（Ingestion Pipeline）

```
输入内容（对话/文档/笔记）
  │
  ├─① Chunker：按语义切分为独立片段
  ├─② Extractor：LLM 提取实体、事实、决策
  ├─③ Summarizer：生成 L0 标签 + L1 摘要
  ├─④ Embedder：对 L1 摘要生成向量
  ├─⑤ TreeIndexer：将新记忆插入树状索引合适位置
  └─⑥ Store：
      · 元数据 + 向量 → DuckDB
      · 原始内容 → Markdown 文件
      · 更新树状索引
```

### 检索流程（Retrieval）— 与场景管理器协同

```
查询输入
  │
  ├─① 先查缓存热区/温区（内存操作，微秒级）
  │   命中 → 直接返回，跳过后续步骤
  │
  ├─② 缓存未命中，双路检索冷区（并行）：
  │   ├─ 向量路：embedding 相似度搜索（DuckDB）
  │   └─ 树状路：LLM 从根节点推理导航
  │
  ├─③ 融合排序：合并去重，综合评分
  │
  ├─④ LOD 加载：
  │   · 先返回 L0/L1（低成本）
  │   · 按需展开 L2（读文件）
  │
  └─⑤ 场景管理器介入（after_model 阶段）：
      · Prefetcher：预测下一步，预加载到温区
      · Frustum Culler：检测话题转换，降级不相关记忆
      · LOD Manager：调整各记忆的精度级别
```

### 缓存管理与遗忘

```
每轮交互后（Scene Manager 自动执行）：
  · 预加载：根据话题方向，从冷区提升相关记忆到温区
  · 降级：不再相关的记忆从热区降到温区/冷区
  · 遗忘：活跃度低于阈值的记忆压缩或归档

活跃度 = importance × e^(-λ × days_since_access) × log(access_count + 1)
```

## SDK 接口设计

### 设计原则

MemoryAtlas 实现为 LangChain 1.0 的 `AgentMiddleware` 子类。
agent 开发者只需要把它加到 middleware 列表里，
记忆的写入、检索、预加载、剔除、遗忘全部自动完成。

### LangChain 1.0 Middleware 钩子映射

| LangChain 钩子 | MemoryAtlas 用途 | 场景管理器动作 |
|---|---|---|
| `before_agent` | 初始化会话缓存 | 加载用户画像到热区 |
| `before_model` | 检索相关记忆注入 messages | 缓存命中优先，LOD 动态选择 |
| `after_model` | 评估写入 + 场景更新 | 预加载 + 视锥剔除 + 精度调整 |
| `after_agent` | 持久化缓存状态 | 记录话题转换模式 |

### 核心用法：一行接入

```python
from langchain.agents import create_agent
from memory_atlas.langchain import MemoryAtlasMiddleware

memory = MemoryAtlasMiddleware(
    storage_path="./my_agent_memory",
    embedding_model="local",
    max_memory_tokens=2000,
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search, code_interpreter],
    middleware=[memory],                # ← 就这一行
)
```

### 中间件内部实现逻辑

```python
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from typing import Any

class MemoryAtlasMiddleware(AgentMiddleware):
    """LangChain 1.0 AgentMiddleware，游戏引擎式记忆管理。"""

    def __init__(self, storage_path: str, **kwargs):
        super().__init__()
        self.engine = MemoryEngine(storage_path, **kwargs)
        self.scene = SceneManager(self.engine)

    def before_agent(self, state, runtime) -> dict[str, Any] | None:
        """加载用户画像、长期记忆索引到缓存。"""
        self.scene.initialize_session(runtime.context)
        return None

    def before_model(self, state, runtime) -> dict[str, Any] | None:
        """检索相关记忆，优先从缓存命中。"""
        query = state["messages"][-1].content

        # 场景管理器：先查热区/温区，再查冷区
        memories = self.scene.get_memory_view(query)

        if memories:
            context = self.engine.format_memories(memories)
            from langchain.messages import SystemMessage
            return {"messages": [SystemMessage(content=context)]}
        return None

    def after_model(self, state, runtime) -> dict[str, Any] | None:
        """评估写入 + 场景管理器更新。"""
        recent = self._extract_recent_turn(state["messages"])

        # 自动判断是否写入
        self.engine.maybe_ingest(recent)

        # 核心差异化：场景管理器更新
        self.scene.update(
            recent_turn=recent,
            # 预加载下一步可能需要的记忆
            # 剔除不再相关的记忆
            # 调整各记忆的 LOD 精度
        )
        return None

    def after_agent(self, state, runtime) -> dict[str, Any] | None:
        """持久化 + 学习话题转换模式。"""
        self.scene.persist()
        self.scene.learn_transition_patterns()
        return None
```

### 自动写入判断

`after_model` 不是每轮都写入。内部用 LLM 快速评估或规则判断：

```python
memory = MemoryAtlasMiddleware(
    storage_path="./memory",
    # LLM 评估模式（默认）
    ingest_strategy="llm",
    ingest_prompt="只记忆与用户项目直接相关的技术决策和 bug 记录",
    # 或规则模式（更快更省钱）
    # ingest_strategy="rule_based",
    # ingest_rules={"min_length": 50, "require_entities": True},
)
```

### 高级用法

```python
# 手动接口（导入历史数据、调试）
from memory_atlas import MemoryEngine

engine = MemoryEngine(storage_path="./my_agent_memory")
engine.bulk_ingest(conversations=[...])
results = engine.retrieve("登录 bug", top_k=5)
detail = engine.expand(memory_id="c001")

# 查看缓存状态
stats = engine.stats()
# {"total": 342, "hot": 12, "warm": 45, "cold": 285,
#  "cache_hit_rate": 0.73, "avg_prefetch_accuracy": 0.61}
```

## 项目源码结构

```
memory-atlas/
├── pyproject.toml
├── README.md
├── LICENSE                      ← MIT
├── docs/
│   ├── DESIGN.md
│   └── PROGRESS.md
├── src/
│   └── memory_atlas/
│       ├── __init__.py
│       ├── engine.py            ← MemoryEngine 核心引擎
│       ├── config.py            ← 配置管理
│       ├── scene/               ← 场景管理器（核心差异化）
│       │   ├── manager.py       ← SceneManager 主逻辑
│       │   ├── prefetcher.py    ← 预测性预加载
│       │   ├── culler.py        ← 视锥剔除
│       │   └── lod.py           ← LOD 精度管理
│       ├── core/
│       │   ├── registry.py      ← DuckDB 元数据管理
│       │   └── tree_index.py    ← 树状索引（PageIndex 风格）
│       ├── ingestion/
│       │   ├── chunker.py       ← 对话分段
│       │   ├── extractor.py     ← LLM 信息提取
│       │   └── summarizer.py    ← 多级摘要生成
│       ├── retrieval/
│       │   ├── vector_search.py ← 向量检索（DuckDB）
│       │   ├── tree_search.py   ← 树状推理检索
│       │   └── fusion.py        ← 双路融合排序
│       ├── storage/
│       │   ├── file_store.py    ← Markdown 文件读写
│       │   └── cache.py         ← 三层缓存（热/温/冷）
│       ├── llm/
│       │   ├── provider.py      ← LiteLLM 封装
│       │   └── embedder.py      ← Embedding（本地 + 云端）
│       └── langchain/
│           ├── __init__.py
│           └── middleware.py    ← MemoryAtlasMiddleware
├── tests/
├── benchmarks/                  ← 性能基准测试（证明差异化价值）
│   ├── cache_hit_rate.py        ← 缓存命中率 vs 纯检索
│   ├── prefetch_accuracy.py     ← 预加载准确率
│   ├── token_savings.py         ← LOD 节省的 token 量
│   └── latency_comparison.py    ← 响应延迟对比
└── examples/
    └── langchain_agent.py
```

## 文件存储结构（运行时数据目录）

```
my_agent_memory/                    ← storage_path
├── index.duckdb                    ← 元数据 + 向量索引
├── tree_index.json                 ← 树状语义索引
├── config.json                     ← 配置
├── chunks/                         ← L2 原始记忆
│   ├── c001.md
│   ├── c002.md
│   └── ...
└── logs/                           ← 操作日志（可选）
    └── ingestion.log
```

## 需要量化证明的指标

MemoryAtlas 的差异化价值需要用数据说话：

| 指标 | 含义 | 目标 |
|---|---|---|
| 缓存命中率 | 热区/温区命中 vs 冷区检索的比例 | > 60% |
| 预加载准确率 | 预加载的记忆在下一轮被实际使用的比例 | > 50% |
| Token 节省率 | LOD 机制相比全量注入节省的 token | > 40% |
| 检索延迟 | 缓存命中 vs 冷区检索的延迟差异 | 缓存 < 10ms, 冷区 < 200ms |
| 上下文相关性 | 注入的记忆与当前对话的相关性评分 | 视锥剔除后 > 0.8 |
