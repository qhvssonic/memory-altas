# MemoryAtlas — 开发进度

> 最后更新：2026-03-26

## 项目状态：🟢 v0.1.0 MVP 完成

## 核心差异化：游戏引擎式场景管理

MemoryAtlas 的核心卖点是 Scene Manager（场景管理器），这是目前所有竞品都没有的。
开发优先级应围绕场景管理器展开，其他模块（写入管道、检索引擎）是基础设施。

## 里程碑

### v0.1.0 — MVP（最小可用版本）

目标：完成所有核心功能，可作为 SDK 被 LangChain 1.0 agent 集成。

| 模块 | 任务 | 状态 | 优先级 | 备注 |
|---|---|---|---|---|
| **项目基础** | 项目结构搭建 | ✅ 完成 | P0 | pyproject.toml, src 结构, LICENSE |
| | README.md | ✅ 完成 | P0 | 突出游戏引擎灵感和差异化 |
| | CI/CD 配置 | ✅ 完成 | P1 | GitHub Actions (Python 3.10/3.11/3.12) |
| **存储层** | DuckDB 元数据管理 (registry) | ✅ 完成 | P0 | 6张表 CRUD + 向量搜索 + 实体/树节点/话题转换 |
| | Markdown 文件存储 (file_store) | ✅ 完成 | P0 | MemoryChunk 序列化/反序列化, frontmatter 解析 |
| | 三层缓存 (cache) | ✅ 完成 | P0 | Hot(dict)/Warm(LRU)/Cold, 升降级, 容量管理 |
| **⭐ 场景管理器** | SceneManager 主逻辑 | ✅ 完成 | **P0** | 编排 prefetch/cull/LOD, 缓存优先检索 |
| | Prefetcher 预测性预加载 | ✅ 完成 | **P0** | 话题转换历史 + 实体扩展 + LLM 预测三策略 |
| | Frustum Culler 视锥剔除 | ✅ 完成 | **P0** | 显式/隐式信号检测, 实体漂移, idle 计数 |
| | LOD Manager 精度管理 | ✅ 完成 | **P0** | L0/L1/L2 动态切换, token 预算控制 |
| | 话题转换模式学习 | ✅ 完成 | P1 | topic_transitions 表, 增量记录 |
| **写入管道** | 语义分段 (chunker) | ✅ 完成 | P0 | turn/paragraph/fixed 三种策略 |
| | 信息提取 (extractor) | ✅ 完成 | P0 | LLM 提取 + 规则回退 |
| | 多级摘要生成 (summarizer) | ✅ 完成 | P0 | L0 标签 + L1 摘要, LLM + 规则回退 |
| | Embedding 生成 | ✅ 完成 | P0 | 本地 sentence-transformers + 云端 LiteLLM |
| | 树状索引构建 (tree_indexer) | ✅ 完成 | P0 | JSON 树, DFS 查找, 自动归类 |
| **检索引擎** | 向量检索 (vector_search) | ✅ 完成 | P0 | DuckDB list_cosine_similarity |
| | 树状推理检索 (tree_search) | ✅ 完成 | P0 | LLM 导航树状索引, 深度优先 |
| | 双路融合排序 (fusion) | ✅ 完成 | P0 | 加权合并 + 双路命中 20% boost |
| **维护** | 遗忘机制 | ✅ 完成 | P1 | 活跃度衰减 + 压缩/归档, ForgettingManager |
| **LLM 层** | LiteLLM 封装 | ✅ 完成 | P0 | complete + complete_json |
| | Embedding 提供者 | ✅ 完成 | P0 | local/openai/cohere 工厂模式 |
| **SDK 接口** | MemoryAtlasMiddleware | ✅ 完成 | P0 | 4 个钩子: before/after_agent, before/after_model |
| | MemoryEngine 核心引擎 | ✅ 完成 | P0 | ingest/retrieve/expand/stats |
| | 自动写入判断 | ✅ 完成 | P0 | rule_based + llm 两种策略 |
| | 手动接口 | ✅ 完成 | P1 | bulk_ingest / retrieve / expand |
| | 配置管理 | ✅ 完成 | P0 | dataclass + JSON 序列化 |
| **基准测试** | 缓存命中率测试 | ✅ 完成 | P0 | 76% 命中率 (目标>60%) |
| | 预加载准确率测试 | ✅ 完成 | P0 | 100% 准确率 (目标>50%) |
| | Token 节省率测试 | ✅ 完成 | P0 | 93.4% 节省 (目标>40%) |
| | 延迟对比测试 | ✅ 完成 | P1 | 缓存 0.3µs vs 冷区 22ms, 69000x 加速 |
| **测试** | 单元测试 | ✅ 完成 | P0 | 72 tests, 11 文件, uv run pytest |
| | 集成测试 | ✅ 完成 | P1 | 6 个端到端测试 (FakeEmbedder/FakeLLM) |
| **文档** | API 文档 | ✅ 完成 | P1 | docs/API.md |
| | 使用示例 | ✅ 完成 | P0 | examples/langchain_agent.py |

### v0.2.0 — 增强版

目标：多用户支持、记忆簇、导入导出、更多 embedding 模型。

| 模块 | 任务 | 状态 | 优先级 | 备注 |
|---|---|---|---|---|
| **记忆导入/导出** | JSON 导出 | ✅ 完成 | P0 | 全量/按 session, 含 L2 内容 |
| | JSON 导入 | ✅ 完成 | P0 | merge/overwrite 两种模式 |
| **多用户支持** | user_id 隔离 | ✅ 完成 | P0 | DuckDB 列 + list_memories 过滤 |
| | 多智能体命名空间 | ✅ 完成 | P1 | agent_id 分区 |
| **记忆簇** | MemoryCluster 数据模型 | ✅ 完成 | P0 | DuckDB 表 + CRUD |
| | 簇的自动聚合 | ✅ 完成 | P1 | 基于实体自动归簇 |
| | 簇级别 LOD | ⬜ 待开始 | P1 | 整簇摘要 |
| **Embedding 扩展** | Ollama 本地模型 | ✅ 完成 | P0 | 通过 LiteLLM |
| | 自定义 Embedder 接口 | ✅ 完成 | P1 | CustomEmbedder(embed_fn, dim) |
| **Web UI** | 记忆图谱可视化 | ⬜ 待开始 | P2 | 独立 HTML，可选 |
| **测试** | v0.2 新功能测试 | ✅ 完成 | P0 | 13 tests |

### v0.3.0 — 生态集成（规划中）

- [ ] 其他框架适配（LlamaIndex、CrewAI 等）
- [ ] MCP Server
- [ ] CLI 工具

## 设计决策记录

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-03-26 | 场景管理器作为核心差异化 | 竞品分析显示预加载/视锥剔除/三层缓存是独有特性 |
| 2026-03-26 | 基准测试纳入 MVP | 差异化价值需要数据证明，不能只靠概念 |
| 2026-03-26 | v0.1.0 核心代码实现完成 | 所有 P0 模块已实现，待测试和基准验证 |
| 2026-03-25 | 项目名定为 MemoryAtlas | Atlas = 纹理图集（游戏引擎）+ 地图集 |
| 2026-03-25 | 使用 DuckDB 而非 PostgreSQL | 嵌入式、零依赖、单文件，适合 SDK 场景 |
| 2026-03-25 | 记忆本体用 Markdown 文件 | 人类可读、Git 友好 |
| 2026-03-25 | 双路检索（向量 + 树状推理） | 向量擅长语义模糊匹配，树状推理擅长逻辑导航 |
| 2026-03-25 | LiteLLM 做 LLM 统一接口 | 不绑定供应商 |
| 2026-03-25 | MIT License | 最宽松，利于社区采用 |
| 2026-03-25 | Python 为主要语言 | LLM 生态最成熟 |
| 2026-03-25 | 首要集成 LangChain 1.0 | Middleware 机制天然适配 |
| 2026-03-25 | 核心逻辑独立为 MemoryEngine | 框架无关，LangChain middleware 只是薄封装 |

## 实现摘要 (2026-03-26)

### 已完成的模块

1. **项目基础**: pyproject.toml (hatch 构建), LICENSE (MIT), README.md
2. **配置管理** (`config.py`): 30+ 配置项, JSON 持久化
3. **存储层**:
   - `core/registry.py`: DuckDB 6 张表 (memories, entities, memory_entities, memory_links, tree_nodes, topic_transitions), 向量搜索, 实体管理, 话题转换记录
   - `storage/file_store.py`: Markdown frontmatter 序列化, chunks 目录管理
   - `storage/cache.py`: Hot(dict)/Warm(OrderedDict LRU)/Cold 三层, 升降级, 容量淘汰
4. **LLM 层**:
   - `llm/provider.py`: LiteLLM 封装, complete + complete_json
   - `llm/embedder.py`: local(sentence-transformers)/openai/cohere 工厂
5. **写入管道**:
   - `ingestion/chunker.py`: turn/paragraph/fixed 三种分段策略
   - `ingestion/extractor.py`: LLM 实体/事实/决策提取 + 规则回退
   - `ingestion/summarizer.py`: L0 标签 + L1 摘要生成
6. **检索引擎**:
   - `retrieval/vector_search.py`: DuckDB cosine similarity
   - `retrieval/tree_search.py`: LLM 导航树状索引
   - `retrieval/fusion.py`: 加权融合 + 双路 boost
7. **⭐ 场景管理器** (核心差异化):
   - `scene/manager.py`: 编排所有子系统, 缓存优先检索, 会话管理
   - `scene/prefetcher.py`: 话题转换历史 + 实体扩展 + LLM 预测
   - `scene/culler.py`: 显式/隐式话题转换检测, 实体漂移, idle 降级
   - `scene/lod.py`: L0/L1/L2 动态切换, token 预算控制
8. **核心引擎** (`engine.py`): 完整 ingest/retrieve/expand 流程
9. **LangChain 集成** (`langchain/middleware.py`): 4 钩子 AgentMiddleware
10. **示例** (`examples/langchain_agent.py`): 独立使用 + LangChain 集成

### 待完成

v0.1.0 所有任务已完成，进入 v0.2.0 规划阶段。

## 灵感来源

- 游戏引擎资源管理（LOD、流式加载、空间分区、视锥剔除）
- [DuckDB](https://duckdb.org/) — 嵌入式分析数据库
- [PageIndex](https://github.com/VectifyAI/PageIndex) — 无向量、基于推理的 RAG 框架
- OpenClaw — 文件系统存储记忆的实践
