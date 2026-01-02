# IntelliFlow Demo (FastAPI + RAG + Agents + MCP)

最小可运行示例，涵盖：FastAPI、pgvector RAG、会话记忆、代理编排、SSE、MCP 工具（GitHub 发布 + 本地文件写入）、APScheduler 定时任务。

## 快速开始
1. 安装依赖  
   ```bash
   pip install -r requirements.txt
   ```
2. 配置环境变量（可复制 `.env.example`）：  
   - `DATABASE_URL`：Postgres，需启用 pgvector 扩展。  
   - `OPENAI_API_KEY`：用于嵌入和对话。  
   - `GITHUB_TOKEN`/`GITHUB_REPO`：可选，启用 GitHub 发布。  
3. 初始化 DB 表：启动服务时自动建表。
4. 导入示例数据：  
   ```bash
   python -m scripts.ingest_sample
   ```
5. 运行 API：  
   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## 主要端点
- `GET /health`：健康检查  
- `POST /rag/query`：RAG 问答（`stream=true` 开启 SSE）  
- `POST /agents/run`：代理编排，支持发布（GitHub/local，`stream=true` 开启 SSE）  
- `GET /agents/memory/{session_id}`：查看最近窗口与摘要

示例请求（代理发布到本地）：  
```bash
curl -X POST http://localhost:8000/agents/run \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "写一段关于IntelliFlow的简短介绍，Markdown输出",
    "session_id": "demo",
    "publish": {"mode": "local", "path": "demo/output.md"}
  }'
```

## 架构要点
- RAG：Tika 解析 → 结构感知/Token 分块 → OpenAI 向量 → pgvector 粗排 → Cohere Rerank 精排。  
- Memory：短期窗口（内存/Redis），超长后 LLM 自动摘要入库并向量化。  
- 代理：聚合上下文 + 记忆 → LLM 规划 → MCP 工具（GitHub、本地文件）→ SSE 事件。  
- 调度：APScheduler 定时运行示例任务。

## Rerank 配置
本系统集成了一套标准的两阶段检索流程：
1. **粗排**：pgvector 快速检索出 Top-N 个候选块。
2. **精排**：调用 Cohere Rerank 模型对候选块进行深度对比重新打分。

在 `.env` 中配置 `COHERE_API_KEY` 即可开启精排功能。若未配置，系统将自动降级为纯向量检索。

## 运行需求
- Postgres (pgvector 已安装)  
- Python 3.10+  
- 可选：Java 环境用于 `tika`（若使用 Tika 解析），否则可自备纯文本/Markdown。

## 测试
- 默认跳过集成测试；设置 `RUN_DB_TESTS=1` 且提供可用 `DATABASE_URL` 后运行：
  ```bash
  RUN_DB_TESTS=1 pytest -q
  ```


uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


```shell
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://learn.microsoft.com/zh-cn/azure/architecture/ai-ml/guide/rag/rag-chunking-phase",
    "source": "microsoft_blog",
    "chunk_size": 800,
    "chunk_overlap": 100
  }'
```
```shell
# 查看最近导入的 5 个分块及其所属文档来源
psql intelliflow -c "
SELECT 
    d.source, 
    c.chunk_index, 
    LEFT(c.content, 100) as content_preview, 
    c.created_at 
FROM document_chunks c 
JOIN documents d ON c.document_id = d.id 
ORDER BY c.created_at DESC 
LIMIT 5;"
```

```shell
curl -X POST "http://localhost:8000/ingest/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@assets/watermark.pdf"
```
```
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "你好你给我讲一下LLM watermark",
    "top_k": 3
  }'
```