from typing import List

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.rag import advisors, retriever
from app.schemas import QueryRequest, QueryResponse, RetrievedChunk
from app.services import llm

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    contexts = advisors.gather_context(req.query, top_k=req.top_k)
    chunks: List[RetrievedChunk] = []
    for c in retriever.retrieve(req.query, top_k=req.top_k):
        chunks.append(c)

    messages = advisors.build_system_prompt(req.query, contexts)
    messages.append({"role": "user", "content": req.query})

    if req.stream:
        def event_gen():
            yield f"event: context\ndata: {len(contexts)}\n\n"
            for chunk in llm.chat_stream(messages):
                yield f"event: llm\ndata: {chunk}\n\n"
            yield "event: done\ndata: complete\n\n"

        return EventSourceResponse(event_gen())

    answer = llm.chat(messages)
    return QueryResponse(query=req.query, answers=[answer], contexts=chunks)

