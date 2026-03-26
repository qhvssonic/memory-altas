# MemoryAtlas

> 第一个用游戏引擎资源管理思想设计的 AI Agent 记忆系统

<p align="center">
  <a href="#安装">安装</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#核心思想">核心思想</a> •
  <a href="#cli-工具">CLI</a> •
  <a href="#基准测试">基准测试</a> •
  <a href="docs/API.md">API 文档</a>
</p>

---

## 为什么需要 MemoryAtlas？

现有记忆系统（Mem0、Zep、Letta）都是**被动检索**——query 来了，搜索，返回结果。

MemoryAtlas 引入游戏引擎的**主动场景管理**——像游戏引擎管理纹理和模型一样，主动预测、预加载、分级、剔除记忆，让 agent 在任意时刻都拥有最优的记忆视野。

| 特性 | Mem0 | Letta | Zep | MemoryAtlas |
|---|:---:|:---:|:---:|:---:|
| 预测性预加载 | ✗ | ✗ | ✗ | ✓ |
| 视锥剔除（主动遗忘） | ✗ | ✗ | ✗ | ✓ |
| 多级精度 LOD | ✗ | 部分 | ✗ | ✓ |
| 三层缓存 | ✗ | ✗ | ✗ | ✓ |
| 记忆簇（Asset Bundling） | ✗ | ✗ | ✗ | ✓ |
| 零外部依赖 | ✗ | ✗ | ✗ | ✓ |
| 本地存储，Git 友好 | ✗ | ✗ | ✗ | ✓ |

## 安装

```bash
pip install memory-atlas

# 本地 embedding（离线可用）：
pip install memory-atlas[local-embedding]
```

## 快速开始

### 一行接入 LangChain

```python
from memory_atlas.langchain import MemoryAtlasMiddleware

memory = MemoryAtlasMiddleware(
    storage_path="./my_agent_memory",
    embedding_model="local",
    max_memory_tokens=2000,
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[memory],  # 就这一行
)
```

### 独立使用

```python
from memory_atlas import MemoryEngine

engine = MemoryEngine(storage_path="./memory", embedding_model="local")

# 写入
engine.ingest("JWT refresh token 存在竞态条件，决定使用滑动窗口策略。")

# 检索（场景管理器自动处理缓存、LOD、预加载）
memories = engine.retrieve("token 过期问题")
for m in memories:
    print(f"[{m.lod}] {m.display_text}")

# 展开到完整内容
detail = engine.expand(memories[0].id)

# 导出 / 导入
engine.export_memories("backup.json")
engine.import_memories("backup.json", mode="merge")

engine.close()
```

### LlamaIndex / CrewAI

```python
# LlamaIndex
from memory_atlas.integrations.llamaindex import MemoryAtlasRetriever
retriever = MemoryAtlasRetriever(storage_path="./memory")
nodes = retriever.retrieve("auth token bug")

# CrewAI
from memory_atlas.integrations.crewai import MemoryAtlasTool
tool = MemoryAtlasTool(storage_path="./memory")
result = tool.run("search for auth bugs")
```

## CLI 工具

```bash
memory-atlas init -s ./my_memory          # 初始化记忆库
memory-atlas ingest "JWT 有竞态条件" -s ./my_memory  # 写入记忆
memory-atlas search "token 过期" -s ./my_memory      # 搜索记忆
memory-atlas stats -s ./my_memory                    # 查看统计
memory-atlas forget -s ./my_memory                   # 遗忘低活跃记忆
memory-atlas export backup.json -s ./my_memory       # 导出
memory-atlas import backup.json -s ./my_memory       # 导入
memory-atlas clusters -s ./my_memory                 # 查看记忆簇
```

## 核心思想

游戏引擎面对的"世界很大、视野有限"的问题，和 agent 面对的"记忆很多、上下文有限"的问题，本质上是同一个问题。

```
┌─────────────────────────────────────────────────────────────┐
│                    Scene Manager（场景管理器）                 │
│                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │  Prefetcher   │  │  LOD Manager  │  │ Frustum Culler │  │
│  │               │  │               │  │                │  │
│  │  根据对话方向  │  │  远处用 L0    │  │  话题转换时     │  │
│  │  预测下一步    │  │  近处用 L1    │  │  主动卸载       │  │
│  │  需要的记忆    │  │  聚焦用 L2    │  │  不相关记忆     │  │
│  └───────┬───────┘  └───────┬───────┘  └───────┬────────┘  │
│          └──────────────────┼──────────────────┘           │
│                             ▼                               │
│              Memory View（当前最优记忆视野）                  │
└─────────────────────────────────────────────────────────────┘
```

| 游戏引擎概念 | MemoryAtlas 对应 |
|---|---|
| LOD（Level of Detail） | L0 标签 / L1 摘要 / L2 完整内容 |
| 流式预加载（Prefetch） | 预测下一轮话题，提前加载到温区 |
| 视锥剔除（Frustum Culling） | 检测话题转换，主动降级不相关记忆 |
| 资产打包（Asset Bundling） | 记忆簇，相关记忆整簇加载/卸载 |
| 分层缓存 | 热区(内存) → 温区(LRU) → 冷区(DuckDB+文件) |

### 三层缓存

```
  热区（Hot）  ← 当前上下文，O(1) 访问，~0.3µs
  温区（Warm） ← 预加载候选 + 最近降级，LRU
  冷区（Cold） ← DuckDB 向量搜索 + 树状推理，~22ms
```

### 记忆簇（自动归簇）

当某个实体关联的记忆数达到阈值，自动打包成簇：

```python
engine.ingest("JWT token 过期 bug")       # jwt → 1 条
engine.ingest("JWT refresh token 竞态")   # jwt → 2 条
engine.ingest("JWT 滑动窗口方案")          # jwt → 3 条 → 自动创建 cluster:jwt
```

### 遗忘机制

```
activity = importance × e^(-λ × days) × log(access_count + 2)
```

活跃度低于阈值的记忆自动压缩（删 L2 保 L1）或归档（只保留 L0 标签）。

## 架构

```
┌──────────────────────────────────────────────────────────────┐
│                      MemoryAtlas SDK                         │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │           Scene Manager（核心差异化）                    │  │
│  │  Prefetcher ── LOD Manager ── Frustum Culler           │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│  ┌────────────┐  ┌────────┴───────┐  ┌─────────────────┐    │
│  │ Ingestion  │  │   Retrieval    │  │  Cache Manager  │    │
│  │ Pipeline   │  │   Engine       │  │  Hot/Warm/Cold  │    │
│  │ chunk/     │  │ vector+tree    │  │                 │    │
│  │ extract/   │  │ → fusion       │  │  Cluster Mgr    │    │
│  │ summarize  │  │                │  │  (auto-bundle)  │    │
│  └────────────┘  └────────────────┘  └─────────────────┘    │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  Storage: DuckDB (index) + Markdown (L2 content)      │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  Integrations: LangChain │ LlamaIndex │ CrewAI │ CLI  │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## 基准测试

| 指标 | 结果 | 目标 |
|---|---|---|
| 缓存命中率 | **76%** | > 60% ✅ |
| 预加载准确率 | **100%** | > 50% ✅ |
| Token 节省率（LOD） | **93.4%** | > 40% ✅ |
| 缓存检索延迟 | **0.3µs** | < 10ms ✅ |
| 冷区检索延迟 | **22ms** | < 200ms ✅ |
| 缓存加速比 | **69,000x** | — |

```bash
uv run python -m benchmarks.cache_hit_rate
uv run python -m benchmarks.prefetch_accuracy
uv run python -m benchmarks.token_savings
uv run python -m benchmarks.latency_comparison
```

## 技术选型

| 组件 | 选型 | 理由 |
|---|---|---|
| 元数据索引 | DuckDB | 嵌入式、零依赖、支持向量运算 |
| 记忆存储 | Markdown | 人类可读、Git 友好 |
| LLM 接口 | LiteLLM | 统一接口，不绑定供应商 |
| Embedding | sentence-transformers / OpenAI / Ollama | 本地+云端双模式 |
| CLI | Typer | 类型安全、自动生成帮助 |
| 包管理 | uv | 快 |

## 项目结构

```
src/memory_atlas/
├── engine.py              # 核心引擎
├── config.py              # 配置管理
├── cli.py                 # CLI 工具
├── scene/                 # ⭐ 场景管理器
│   ├── manager.py         #   编排预加载/剔除/LOD
│   ├── prefetcher.py      #   预测性预加载
│   ├── culler.py          #   视锥剔除
│   └── lod.py             #   精度管理
├── core/
│   ├── registry.py        #   DuckDB 元数据
│   ├── tree_index.py      #   树状语义索引
│   └── cluster.py         #   记忆簇
├── ingestion/             # 写入管道
├── retrieval/             # 检索引擎（向量+树+融合）
├── storage/               # 文件存储 + 三层缓存
├── maintenance/           # 遗忘机制
├── llm/                   # LLM + Embedding
├── langchain/             # LangChain 集成
└── integrations/          # LlamaIndex / CrewAI
```

## 开发

```bash
# 安装依赖
uv sync --dev

# 跑测试（91 个）
uv run python -m pytest tests/ -v

# Lint
uv run ruff check src/ tests/

# 跑基准测试
uv run python -m benchmarks.cache_hit_rate
```

## License

MIT
