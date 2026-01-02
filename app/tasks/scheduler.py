import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.agents.orchestrator import AgentOrchestrator
from app.schemas import AgentRequest, PublishConfig

logger = logging.getLogger(__name__)
_scheduler: AsyncIOScheduler | None = None
_orchestrator = AgentOrchestrator()


def init_scheduler(app=None):
    global _scheduler
    if _scheduler:
        return _scheduler

    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(run_sample_job, "interval", minutes=60, id="sample_publish_job")
    _scheduler.start()
    logger.info("Scheduler started with sample job")
    return _scheduler


def run_sample_job():
    prompt = "生成一段今日技术热点的Markdown摘要，100-150字。"
    req = AgentRequest(
        prompt=prompt,
        session_id="scheduled",
        publish=PublishConfig(mode="local", path=f"scheduled/{datetime.utcnow().date()}.md"),
    )
    try:
        result = _orchestrator.run(req)
        logger.info("Scheduled job completed: %s", result.publish_path)
    except Exception as exc:
        logger.warning("Scheduled job failed: %s", exc)

