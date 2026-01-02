from typing import List

from app.memory.summary import search_summaries
from app.schemas import RetrievedChunk
from app.rag import retriever


def gather_context(query: str, top_k: int | None = None) -> List[str]:
    """
    Collect context from RAG chunks + memory summaries.
    """
    contexts: List[str] = []
    chunks = retriever.retrieve(query, top_k=top_k)
    contexts.extend([c.content for c in chunks])

    summaries = search_summaries(query, top_k=2)
    contexts.extend([s.content for s in summaries])
    return contexts


def format_context(contexts: List[str]) -> str:
    formatted = []
    for idx, ctx in enumerate(contexts):
        formatted.append(f"[{idx+1}] {ctx}")
    return "\n".join(formatted)


def build_system_prompt(query: str, contexts: List[str]) -> List[dict]:
    prompt = (
        "你是负责知识检索和工具调度的智能助手。"
        "请用提供的上下文回答用户问题，必要时提出行动计划。"
        "如果需要发布文章，会在计划中写出摘要和输出路径。"
    )
    messages = [{"role": "system", "content": prompt}]
    context_text = format_context(contexts)
    if context_text:
        messages.append({"role": "system", "content": f"上下文:\n{context_text}"})
    return messages

