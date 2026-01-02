import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db import Base, engine, init_pgvector_extension, create_vector_indexes
from app.routers import rag, agents, ingest
from app.schemas import HealthResponse
from app.tasks.scheduler import init_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, debug=settings.debug)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse()

    app.include_router(rag.router, prefix="/rag", tags=["rag"])
    app.include_router(agents.router, prefix="/agents", tags=["agents"])
    app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])

    @app.on_event("startup")
    async def on_startup():
        if settings.auto_init_db:
            # 开发/Demo 模式：自动初始化数据库
            logger.info("Auto-initializing database (set AUTO_INIT_DB=false for production)...")
            init_pgvector_extension()
            logger.info("pgvector extension ready")
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created")
            try:
                create_vector_indexes()
                logger.info("Vector indexes created")
            except Exception as exc:
                logger.warning("Vector index creation deferred: %s", exc)
        else:
            # 生产模式：假设数据库已通过 Alembic 迁移就绪
            logger.info("Skipping auto DB init (production mode)")
        init_scheduler(app)

    return app


app = create_app()

