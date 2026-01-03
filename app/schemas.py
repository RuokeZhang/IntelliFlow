from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    session_id: str | None = None
    stream: bool = False


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_index: int
    content: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answers: List[str]
    contexts: List[RetrievedChunk]


class PublishConfig(BaseModel):
    mode: str = Field(default="github", description="github | local")
    path: str = Field(default="content/output.md")
    branch: Optional[str] = None


class AgentRequest(BaseModel):
    prompt: str
    session_id: str
    stream: bool = False
    publish: Optional[PublishConfig] = None
    top_k: int | None = None


class AgentResult(BaseModel):
    session_id: str
    output: str
    publish_path: Optional[str] = None
    publish_url: Optional[str] = None


class MemoryMessage(BaseModel):
    type: str
    role: Optional[str] = None
    tool_name: Optional[str] = None
    content: str


class MemorySummaryItem(BaseModel):
    id: str
    content: str
    created_at: str


class MemoryDebugResponse(BaseModel):
    session_id: str
    recent: List[MemoryMessage]
    summaries: List[MemorySummaryItem]


class IngestResponse(BaseModel):
    document_id: str
    chunks: int


class UrlIngestRequest(BaseModel):
    url: str
    source: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


class ToolDecision(BaseModel):
    """
    LLM 输出的工具调用决策（结构化输出）。
    用于意图识别后决定是否发布、发布到哪里。
    """
    action: str = Field(
        description="决策动作: 'none'=不发布, 'publish'=执行发布, 'ask_destination'=询问用户发布位置"
    )
    destination: Optional[str] = Field(
        default=None,
        description="发布目标: 'local' | 'github' | null"
    )
    path: Optional[str] = Field(
        default=None,
        description="发布路径，如 'output.md' 或 'content/article.md'"
    )
    reason: Optional[str] = Field(
        default=None,
        description="决策理由（用于调试/日志）"
    )
    pending_content: Optional[str] = Field(
        default=None,
        description="待发布的完整内容（当 action='ask_destination' 时暂存，等用户确认后发布）"
    )

