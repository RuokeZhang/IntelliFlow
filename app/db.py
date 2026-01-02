from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from app.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    echo=settings.debug,
    pool_size=settings.database_pool_size,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_pgvector_extension() -> None:
    """
    Ensure pgvector extension exists before creating tables.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def create_vector_indexes() -> None:
    """
    Create vector indexes after tables are created.
    Uses HNSW index (pgvector 0.5+) which works on empty tables.
    Falls back gracefully if index already exists.
    """
    with engine.begin() as conn:
        # HNSW 索引：可在空表创建，查询性能好，适合 demo/中小规模
        # 生产环境大规模数据可考虑 ivfflat 或分区
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding "
                "ON document_chunks USING hnsw (embedding vector_l2_ops)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_memory_summaries_embedding "
                "ON memory_summaries USING hnsw (embedding vector_l2_ops)"
            )
        )

