# MemoryAtlas — 开发进度

> 最后更新：2026-03-26

## 项目状态：🟡 设计阶段

## 核心差异化：游戏引擎式场景管理

MemoryAtlas 的核心卖点是 Scene Manager（场景管理器），这是目前所有竞品都没有的。
开发优先级应围绕场景管理器展开，其他模块（写入管道、检索引擎）是基础设施。

## 里程碑

### v0.1.0 — MVP（最小可用版本）

目标：完成所有核心功能，可作为 SDK 被 LangChain 1.0 agent 集成。

| 模块 | 任务 | 状态 | 优先级 | 备注 |
|---|---|---|---|---|
| **项目基础** | 项目结构搭建 | ⬜ 待开始 | P0 | pyproject.toml, src 结构 |
| | README.md | ⬜ 待开始 | P0 | 突出游戏引擎灵感和差异化 |
| | CI/CD 配置 | ⬜ 待开始 | P1 | GitHub Actions |
| **存储层** | DuckDB 元数据管理 (registry) | ⬜ 待开始 | P0 | 表结构、CRUD |
| | Markdown 文件存储 (file_store) | ⬜ 待开始 | P0 | 读写 chunks/*.md |
| | 三层缓存 (cache) | ⬜ 待开始 | P0 | 热/温/冷区，LRU |
| **⭐ 场景管理器** | SceneManager 主逻辑 | ⬜ 待开始 | **P0** | **核心差异化** |
| | Prefetcher 预测性预加载 | ⬜ 待开始 | **P0** | 话题关联 + 实体扩展 |
| | Frustum Culler 视锥剔除 | ⬜ 待开始 | **P0** | 话题转换检测 + 主动降级 |
| | LOD Manager 精度管理 | ⬜ 待开始 | **P0** | L0/L1/L2 动态切换 |
| | 话题转换模式学习 | ⬜ 待开始 | P1 | topic_transitions 表 |
| **写入管道** | 语义分段 (chunker) | ⬜ 待开始 | P0 | 按话题切分对话 |
| | 信息提取 (extractor) | ⬜ 待开始 | P0 | LLM 提取实体/事实/决策 |
| | 多级摘要生成 (summarizer) | ⬜ 待开始 | P0 | L0 标签 + L1 摘要 |
| | Embedding 生成 | ⬜ 待开始 | P0 | 本地 + 云端双模式 |
| | 树状索引构建 (tree_indexer) | ⬜ 待开始 | P0 | 新记忆插入合适位置 |
| **检索引擎** | 向量检索 (vector_search) | ⬜ 待开始 | P0 | DuckDB 向量相似度 |
| | 树状推理检索 (tree_search) | ⬜ 待开始 | P0 | LLM 导航树状索引 |
| | 双路融合排序 (fusion) | ⬜ 待开始 | P0 | 合并去重 + 综合评分 |
| **维护** | 遗忘机制 | ⬜ 待开始 | P1 | 活跃度衰减 + 压缩归档 |
| **LLM 层** | LiteLLM 封装 | ⬜ 待开始 | P0 | 统一接口 |
| | Embedding 提供者 | ⬜ 待开始 | P0 | 本地 sentence-transformers + 云端 |
| **SDK 接口** | MemoryAtlasMiddleware | ⬜ 待开始 | P0 | LangChain 1.0 AgentMiddleware |
| | MemoryEngine 核心引擎 | ⬜ 待开始 | P0 | 框架无关的核心逻辑 |
| | 自动写入判断 | ⬜ 待开始 | P0 | LLM 评估 + 规则策略 |
| | 手动接口 | ⬜ 待开始 | P1 | bulk_ingest / retrieve / expand |
| | 配置管理 | ⬜ 待开始 | P0 | config.json |
| **基准测试** | 缓存命中率测试 | ⬜ 待开始 | P0 | 证明缓存价值 |
| | 预加载准确率测试 | ⬜ 待开始 | P0 | 证明预加载价值 |
| | Token 节省率测试 | ⬜ 待开始 | P0 | 证明 LOD 价值 |
| | 延迟对比测试 | ⬜ 待开始 | P1 | 缓存 vs 冷区 |
| **测试** | 单元测试 | ⬜ 待开始 | P0 | pytest |
| | 集成测试 | ⬜ 待开始 | P1 | 端到端流程 |
| **文档** | API 文档 | ⬜ 待开始 | P1 | |
| | 使用示例 | ⬜ 待开始 | P0 | examples/langchain_agent.py |

### v0.2.0 — 增强版（规划中）

- [ ] 记忆簇（Memory Cluster / Asset Bundling）
- [ ] 多用户/多智能体支持
- [ ] 记忆导入/导出
- [ ] Web UI 可视化记忆图谱
- [ ] 更多 embedding 模型支持

### v0.3.0 — 生态集成（规划中）

- [ ] 其他框架适配（LlamaIndex、CrewAI 等）
- [ ] MCP Server
- [ ] CLI 工具

## 设计决策记录

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-03-26 | 场景管理器作为核心差异化 | 竞品分析显示预加载/视锥剔除/三层缓存是独有特性 |
| 2026-03-26 | 基准测试纳入 MVP | 差异化价值需要数据证明，不能只靠概念 |
| 2026-03-25 | 项目名定为 MemoryAtlas | Atlas = 纹理图集（游戏引擎）+ 地图集 |
| 2026-03-25 | 使用 DuckDB 而非 PostgreSQL | 嵌入式、零依赖、单文件，适合 SDK 场景 |
| 2026-03-25 | 记忆本体用 Markdown 文件 | 人类可读、Git 友好 |
| 2026-03-25 | 双路检索（向量 + 树状推理） | 向量擅长语义模糊匹配，树状推理擅长逻辑导航 |
| 2026-03-25 | LiteLLM 做 LLM 统一接口 | 不绑定供应商 |
| 2026-03-25 | MIT License | 最宽松，利于社区采用 |
| 2026-03-25 | Python 为主要语言 | LLM 生态最成熟 |
| 2026-03-25 | 首要集成 LangChain 1.0 | Middleware 机制天然适配 |
| 2026-03-25 | 核心逻辑独立为 MemoryEngine | 框架无关，LangChain middleware 只是薄封装 |

## 灵感来源

- 游戏引擎资源管理（LOD、流式加载、空间分区、视锥剔除）
- [DuckDB](https://duckdb.org/) — 嵌入式分析数据库
- [PageIndex](https://github.com/VectifyAI/PageIndex) — 无向量、基于推理的 RAG 框架
- OpenClaw — 文件系统存储记忆的实践
