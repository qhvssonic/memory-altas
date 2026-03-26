# 我用游戏引擎的思想，重新设计了 AI Agent 的记忆系统

> MemoryAtlas：第一个用游戏引擎资源管理思想设计的智能体记忆系统

## 一、现有记忆系统的问题

做过 AI Agent 开发的人都知道，agent 需要"记忆"——记住用户说过什么、做过什么决策、有哪些偏好。市面上已经有不少记忆方案：Mem0、Zep、Letta、Hindsight……

但它们有一个共同的工作模式：

```
用户消息 → 生成 query → 搜索记忆库 → 返回 top-k → 注入 prompt
```

这是**被动检索**。每一轮对话都是冷启动，不管上一轮聊了什么，都从零开始搜索。

这带来几个问题：

1. **无法预判**——用户正在聊认证模块，下一步大概率会聊 JWT 或 OAuth，但系统不会提前准备
2. **无法主动遗忘**——用户明确说"换个话题"，旧记忆仍然占着上下文 token
3. **精度单一**——要么返回完整内容浪费 token，要么返回摘要可能丢细节
4. **每次都是全量搜索**——即使上一轮刚查过同样的记忆

有没有一个领域，面对过完全相同的问题，而且已经有了成熟的解法？

有。游戏引擎。

## 二、游戏引擎是怎么解决这个问题的

一个开放世界游戏，地图可能有几十 GB 的资源——模型、贴图、音效、动画。但玩家的显存只有几个 GB，屏幕上同时只能看到一小片区域。

游戏引擎不会把整个世界一次性加载到内存里，它用了一套精密的资源管理系统：

**LOD（Level of Detail）**——远处的山用低模，走近了换高模。不是所有东西都需要最高精度。

**流式预加载（Streaming/Prefetch）**——玩家朝北走，引擎提前加载北边的资源到内存。等玩家真的走到了，资源已经准备好了，零延迟。

**视锥剔除（Frustum Culling）**——玩家背后的东西不渲染。转身了，背后的资源卸载，面前的资源加载。

**分层缓存**——GPU 显存放正在渲染的，内存放即将用到的，硬盘放暂时不需要的。

这套思想，完美映射到 agent 记忆管理上。

## 三、MemoryAtlas 的核心设计

MemoryAtlas 把游戏引擎的资源管理思想搬到了 agent 记忆系统里。核心是一个 **Scene Manager（场景管理器）**，它编排三个子系统：

### 3.1 预测性预加载（Prefetcher）

现有方案：用户说了什么 → 搜索相关记忆（被动）

MemoryAtlas：用户说了什么 → 搜索相关记忆 + **预测下一步可能聊什么 → 提前加载到温区**（主动）

```
用户："我在做用户认证模块"

  传统方案：
    → 搜索"用户认证"相关记忆，返回结果，结束

  MemoryAtlas 额外做的：
    → 预测接下来可能聊：JWT、session、OAuth、密码加密
    → 从冷区预加载这些话题的记忆到温区
    → 下一轮用户真的聊到 JWT 时，记忆已经在温区，零延迟
```

预测策略有三种，可叠加：
- **话题关联**：历史上 A 话题之后经常聊 B，预加载 B
- **实体扩展**：提到"auth 模块"，预加载所有 auth 相关记忆
- **LLM 预测**：用 LLM 预测下一步可能的话题

我们的基准测试显示，预加载准确率可以达到 100%（在话题可预测的场景下）。

### 3.2 视锥剔除（Frustum Culler）

游戏里玩家背后的东西不渲染。MemoryAtlas 主动检测话题转换，卸载不相关记忆。

```
用户："好，认证的事先放一放，我们来看看数据库性能问题"

  传统方案：
    → 新搜索"数据库性能"，但上一轮的 auth 记忆可能还在上下文里占 token

  MemoryAtlas：
    → 检测到明确的话题转换信号
    → 主动将 auth 相关记忆从热区降级到温区
    → 腾出 token 空间给数据库性能相关记忆
    → 如果用户又回到 auth 话题，从温区快速恢复（不用重新搜索冷区）
```

检测信号包括：
- **显式信号**：用户说"换个话题"、"let's move on"
- **隐式信号**：连续 N 轮没有引用某话题的记忆
- **实体漂移**：当前对话涉及的实体与热区记忆的实体重叠度低于阈值

### 3.3 三级精度动态切换（LOD Manager）

不是所有记忆都需要同样的细节。MemoryAtlas 根据相关性动态选择精度：

```
L0（标签）："2024-03 讨论了 auth 模块的 JWT token 过期 bug"
  → ~20 tokens，用于索引和粗筛

L1（摘要）："refresh token 存在竞态条件，决定使用滑动窗口策略"
  → ~80 tokens，大多数场景够用

L2（完整）：原始对话记录，包含所有细节和上下文
  → ~500+ tokens，只在需要精确细节时加载
```

检索到 10 条相关记忆 → 全部以 L0 展示 → 其中 top 20% 自动提升到 L1 → 需要细节时按需展开到 L2。

**总 token 消耗始终可控。** 我们的基准测试显示，LOD 机制相比全量注入节省了 93.4% 的 token。

### 3.4 三层缓存

```
┌─────────────────────────────────────────────────┐
│  热区（Hot）— 当前上下文中正在使用的记忆          │
│  内存字典，O(1) 访问，延迟 ~0.3µs               │
├─────────────────────────────────────────────────┤
│  温区（Warm）— 预加载的候选 + 最近降级的记忆      │
│  内存 LRU 缓存，提升到热区零成本                  │
├─────────────────────────────────────────────────┤
│  冷区（Cold）— 全量存储                          │
│  DuckDB + Markdown 文件，延迟 ~22ms              │
└─────────────────────────────────────────────────┘
```

状态流转：
- 冷 → 温：预加载器根据话题预测提升
- 温 → 热：检索命中时提升
- 热 → 温：视锥剔除器检测到话题转换时降级
- 温 → 冷：LRU 淘汰或长时间未访问

基准测试显示缓存命中率 76%，缓存比冷区检索快 69000 倍。

## 四、概念映射总览

| 游戏引擎概念 | MemoryAtlas 对应 | 竞品是否有 |
|---|---|---|
| LOD（Level of Detail） | L0/L1/L2 三级记忆精度 | Letta 有部分 |
| 流式预加载（Prefetch） | 预测性预加载到温区 | **无** |
| 视锥剔除（Frustum Culling） | 主动排除不相关记忆 | **无** |
| 资产注册表（Asset Registry） | DuckDB 元数据索引 | Mem0 有类似 |
| 资产打包（Asset Bundling） | 记忆簇（Memory Cluster） | **无** |
| Chunk Loading/Unloading | 热/温/冷三层缓存 | **无** |
| 场景管理器（Scene Manager） | 记忆视野管理 | **无** |

## 五、技术实现

### 技术选型

- **Python 3.10+**，uv 管理依赖
- **DuckDB** 做元数据索引和向量搜索——嵌入式、零外部依赖、单文件
- **Markdown 文件** 存记忆本体——人类可读、Git 友好
- **LiteLLM** 统一 LLM 接口——不绑定供应商
- **sentence-transformers** 本地 embedding——离线可用

整个系统 `pip install` 即可使用，不需要 Docker、不需要外部数据库、不需要云服务。

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
    tools=[search, code_interpreter],
    middleware=[memory],  # 就这一行
)
```

记忆的写入、检索、预加载、剔除、遗忘全部自动完成。

### CLI 工具

```bash
memory-atlas init -s ./my_memory
memory-atlas ingest "JWT refresh token 有竞态条件" -s ./my_memory
memory-atlas search "token 过期" -s ./my_memory
memory-atlas stats -s ./my_memory
memory-atlas forget -s ./my_memory
memory-atlas export backup.json -s ./my_memory
memory-atlas clusters -s ./my_memory
```

### 记忆簇：自动归簇

当某个实体（比如 "jwt"）关联的记忆数量达到阈值（默认 3 条），系统自动将这些记忆打包成一个簇。后续加载/卸载都是整簇操作，就像游戏引擎的 Asset Bundle。

```python
# 自动发生在 ingest 流程中，无需手动触发
engine.ingest("JWT token 过期 bug")       # jwt 关联 1 条
engine.ingest("JWT refresh token 竞态")   # jwt 关联 2 条
engine.ingest("JWT 滑动窗口方案")          # jwt 关联 3 条 → 自动创建 cluster:jwt
```

## 六、基准测试数据

差异化价值不能只靠概念，要用数据说话：

| 指标 | 结果 | 目标 |
|---|---|---|
| 缓存命中率 | **76%** | > 60% ✅ |
| 预加载准确率 | **100%** | > 50% ✅ |
| Token 节省率（LOD） | **93.4%** | > 40% ✅ |
| 缓存检索延迟 | **0.3µs** | < 10ms ✅ |
| 冷区检索延迟 | **22ms** | < 200ms ✅ |
| 缓存加速比 | **69000x** | — |

## 七、遗忘机制

人会遗忘，agent 也应该。MemoryAtlas 用活跃度衰减公式管理记忆生命周期：

```
activity = importance × e^(-λ × days_since_access) × log(access_count + 2)
```

- 重要性高、经常被访问的记忆活跃度高，长期保留
- 不重要、长期未访问的记忆活跃度衰减
- 低于压缩阈值：删除 L2 原始文件，只保留 L0+L1
- 低于归档阈值：只保留 L0 标签

这让记忆库不会无限膨胀，同时重要的记忆永远不会丢失。

## 八、总结

MemoryAtlas 的核心思想很简单：**游戏引擎面对的"世界很大、视野有限"的问题，和 agent 面对的"记忆很多、上下文有限"的问题，本质上是同一个问题。**

游戏引擎用了几十年打磨出的解法——LOD、预加载、视锥剔除、分层缓存、资源打包——可以直接映射到记忆管理上。

这不是一个理论项目。MemoryAtlas 是一个可用的 Python SDK，91 个测试全绿，支持 LangChain/LlamaIndex/CrewAI，有 CLI 工具，`pip install` 即可使用。

项目地址：[GitHub - MemoryAtlas](https://github.com/qhvssonic/memory-altas)

---

*如果你也在做 AI Agent 开发，欢迎试用和反馈。如果觉得这个思路有意思，点个 star 支持一下。*
