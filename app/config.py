from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    app_name: str = "IntelliFlow"
    debug: bool = True
    database_url: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5432/intelliflow",
        description="Postgres connection string with pgvector extension enabled",
    )
    database_pool_size: int = 5

    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
 
    cohere_api_key: str | None = Field(default=None, description="Cohere API key for Reranking")
    use_rerank: bool = True
    rerank_top_n: int = 10  # 粗排召回数量
 
    github_token: str | None = Field(default=None, description="GitHub PAT with repo scope")
    github_repo: str | None = Field(default=None, description="owner/repo")
    github_branch: str = "main"
    github_base_path: str = "content"

    local_workspace: str = "./workspace"

    # 数据库初始化开关：开发环境 True，生产环境 False（使用 Alembic）
    auto_init_db: bool = Field(default=True, description="启动时自动创建表/索引，生产环境应设为 False")

    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4
    session_window_size: int = 12
    summary_trigger_messages: int = 10

    # Session 配置（生产级）
    redis_url: str | None = Field(default=None, description="Redis URL, e.g. redis://localhost:6379/0")
    session_ttl_seconds: int = Field(default=1800, description="Session 过期时间（秒），默认 30 分钟")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()

