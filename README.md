# knowledge_base

> 面向个人/企业文档的知识库（RAG）服务：文档导入 → 向量化入库 → 混合检索问答，FastAPI + LangGraph 全流程实现。

自然语言提问 → 查询理解与改写 → 稠密+稀疏双向量混合召回（BGE-M3）→ 加权融合 + RRF → 断崖截断 → 生成回答，支持联网检索兜底。

## 核心特性

- **文档导入管线**：PDF/Markdown/图片多格式解析（MinerU + 视觉模型兜底）、主体识别、mistune AST + RecursiveSplitter 结构化切块、BGE-M3 向量化、Milvus 入库，全流程以 LangGraph 子图编排
- **混合检索**：BGE-M3 稠密 + 稀疏双向量、HyDE 查询扩展、加权融合 + RRF 重排、断崖截断（0.5/0.25 阈值）抑制噪声段落
- **查询理解**：前置意图规则 + 查询改写节点，未命中知识库时自动转联网搜索（web_search MCP）
- **服务化**：FastAPI 对外提供导入/查询 API，SSE 流式返回，Mongo 会话历史，MinIO 图片存储

## 技术栈

`Python` · `FastAPI` · `LangGraph` · `Milvus` · `BGE-M3` · `MongoDB` · `MinIO` · `SQLGlot` · `Docker`

## 架构

```text
app/
├── import_process/        # 导入管线（LangGraph 子图）
│   └── agent/nodes/
│       ├── entry/         # 入口与文件预处理
│       ├── parsing/       # pdf→md / md 图片抽取
│       ├── processing/    # 主体识别
│       └── vector/        # BGE-M3 嵌入 + Milvus 入库
├── query/                 # 问答链路（LangGraph 子图）
│   ├── graph/             # pre_search → fusion → generation 子图
│   └── agent/nodes/
│       ├── rewrite/       # 查询理解 + 改写（HyDE）
│       ├── search/        # Milvus 检索 / 联网搜索 MCP
│       └── fusion/        # 加权融合 + RRF
├── core/                  # 节点基类、中间件、装饰器、异常契约
└── lm/                    # LLM / Embedding 客户端封装
```

## 快速开始

```bash
uv sync
cp .env.example .env        # 填入 Milvus / Mongo / MinIO / LLM / MinerU 等配置
python main.py              # 启动服务
```
