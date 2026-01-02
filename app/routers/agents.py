from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.agents.orchestrator import AgentOrchestrator
from app.schemas import AgentRequest, AgentResult, MemoryDebugResponse, MemoryMessage, MemorySummaryItem
from app.memory.summary import list_summaries

router = APIRouter()
orchestrator = AgentOrchestrator()


@router.post("/run", response_model=AgentResult)
async def run_agent(req: AgentRequest):
    if req.stream:
        return EventSourceResponse(orchestrator.run_stream(req))
    return orchestrator.run(req)


@router.post("/run/stream")
async def run_agent_stream(req: AgentRequest):
    return EventSourceResponse(orchestrator.run_stream(req))


@router.get("/memory/{session_id}", response_model=MemoryDebugResponse)
async def debug_memory(session_id: str):
    recent = orchestrator.memory.get_recent(session_id)
    summaries = list_summaries(session_id, limit=5)
    return MemoryDebugResponse(
        session_id=session_id,
        recent=[MemoryMessage(**item) for item in recent],
        summaries=summaries,
    )

