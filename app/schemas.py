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

