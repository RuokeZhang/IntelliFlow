from typing import List, Optional, Tuple
import json
import logging
import re

from app.memory.summary import search_summaries
from app.schemas import RetrievedChunk, ToolDecision
from app.rag import retriever
from app.services import llm


_REWRITE_SYSTEM = (
    "你是一个内部的查询规范化组件，负责把用户的自然语言描述转成对 RAG 检索友好的短句。"
)

logger = logging.getLogger(__name__)


def _rewrite_query_for_retriever(query: str) -> str:
    """
    使用 LLM 对原始 query 进行意图/检索方向的 rewrite。
    如果出错，回退到原文。
    """
    try:
        messages = [
            {"role": "system", "content": _REWRITE_SYSTEM},
            {
                "role": "user",
                "content": (
                    "原始用户 query：\n"
                    f"{query}\n\n"
                    "请提炼成适合检索的简洁句子，仅输出结果。"
                ),
            },
        ]
        response = llm.chat(messages).strip()
        rewritten = response or query
        logger.info("rewrite_query: %s -> %s", query, rewritten)
        return rewritten
    except Exception as exc:
        logger.warning("rewrite_query failed, use original (%s): %s", exc, query)
        return query


def gather_context(query: str, top_k: int | None = None) -> List[str]:
    """
    Collect context from RAG chunks + memory summaries.
    """
    logger.info("gather_context start: original=%s", query)
    contexts: List[str] = []
    search_query = _rewrite_query_for_retriever(query)
    logger.info("gather_context rewrite=%s", search_query)
    chunks = retriever.retrieve(search_query, top_k=top_k)
    contexts.extend([c.content for c in chunks])
    logger.info("gather_context chunks=%d", len(chunks))

    summaries = search_summaries(query, top_k=2)
    contexts.extend([s.content for s in summaries])
    logger.info("gather_context summaries=%d", len(summaries))
    return contexts


def format_context(contexts: List[str]) -> str:
    formatted = []
    for idx, ctx in enumerate(contexts):
        formatted.append(f"[{idx+1}] {ctx}")
    return "\n".join(formatted)


_TOOL_DECISION_SCHEMA = """
你的回复必须严格遵循以下 JSON 格式（放在 ```json 代码块中）：

```json
{
  "answer": "给用户的完整回答内容",
  "tool_decision": {
    "action": "none | publish | ask_destination",
    "destination": "local | github | null",
    "path": "文件路径或null",
    "reason": "决策理由",
    "pending_content": "待发布的完整内容（仅当 action 为 ask_destination 时必填）"
  }
}
```

**action 决策规则（严格遵守）**：
- "none": 用户没有表达发布/保存/导出意图，仅回答问题
- "publish": 用户**明确指定了发布位置**，立即执行发布
- "ask_destination": 用户表达了发布意图，但**没有明确指定位置**，必须追问

**极其重要 - 何时必须使用 ask_destination**：
以下情况必须设置 action="ask_destination"，不能直接发布：
- 用户只说 "publish it" / "save it" / "保存" / "发布" 而没有说 "local" 或 "GitHub"
- 用户说 "write ... and publish" 但没有指定 "to local" 或 "to GitHub"
- 任何没有明确包含 "local"/"本地"/"文件"/"GitHub"/"仓库" 等位置词的发布请求

**何时可以使用 publish（直接发布）**：
只有以下情况才能设置 action="publish"：
- 用户明确说 "save to local" / "保存到本地" / "写入本地文件" → destination="local"
- 用户明确说 "publish to GitHub" / "推送到仓库" / "上传GitHub" → destination="github"
- 用户在上一轮被追问后，回复了 "local" / "本地" / "GitHub" 等位置词

**destination 规则**：
- 用户说"保存到本地"/"写入文件"/"导出到本地"/"local"/"save locally" → "local"
- 用户说"发布到GitHub"/"推送仓库"/"upload to GitHub"/"github" → "github"
- 用户未明确指定 → null（此时 action 必须是 "ask_destination"）

**path 规则**：
- 用户指定了路径则使用用户路径
- 未指定时：本地默认 "output.md"，GitHub 默认 "content/output.md"

**answer 字段规则**：
- 如果 action="ask_destination"：
  1. 先在 answer 中给出完整的生成内容（文章、代码等）
  2. 然后在 answer 末尾追问："您希望保存到本地文件还是发布到 GitHub？"
  3. 同时把生成的内容复制到 pending_content 字段
- 如果 action="publish"：answer 是要发布的完整内容，pending_content 留空
- 如果 action="none"：正常回答问题

**检测用户回复发布位置**：
如果用户的输入只是简单回复"本地"/"local"/"GitHub"等位置词，且对话上下文中有待发布内容，则应该：
- action: "publish"
- destination: 对应位置
- pending_content: 留空（系统会从上下文中获取）
"""


def build_system_prompt(query: str, contexts: List[str], enable_tool_decision: bool = True) -> List[dict]:
    """
    构建系统提示词。
    
    Args:
        query: 用户查询
        contexts: RAG 检索到的上下文
        enable_tool_decision: 是否启用工具调用决策（结构化输出）
    """
    base_prompt = (
        "你是一个智能助手，负责知识检索和工具调度。"
        "请根据提供的上下文回答用户问题。"
    )
    
    if enable_tool_decision:
        base_prompt += _TOOL_DECISION_SCHEMA
    
    messages = [{"role": "system", "content": base_prompt}]
    context_text = format_context(contexts)
    if context_text:
        messages.append({"role": "system", "content": f"上下文:\n{context_text}"})
    return messages


def parse_llm_response(raw_response: str) -> Tuple[str, Optional[ToolDecision]]:
    """
    解析 LLM 的结构化响应，提取 answer 和 tool_decision。
    
    Args:
        raw_response: LLM 原始输出（可能包含 ```json 代码块）
    
    Returns:
        (answer, tool_decision): 用户可见的回答 和 工具调用决策
        如果解析失败，返回 (原始响应, None)
    """
    # 尝试从 ```json ... ``` 代码块中提取
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_response)
    
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            answer = data.get("answer", raw_response)
            
            tool_data = data.get("tool_decision", {})
            if tool_data:
                decision = ToolDecision(
                    action=tool_data.get("action", "none"),
                    destination=tool_data.get("destination"),
                    path=tool_data.get("path"),
                    reason=tool_data.get("reason"),
                    pending_content=tool_data.get("pending_content"),
                )
                logger.info(
                    "parse_llm_response: action=%s, destination=%s, path=%s, reason=%s, has_pending=%s",
                    decision.action, decision.destination, decision.path, decision.reason,
                    bool(decision.pending_content)
                )
                return answer, decision
            
            return answer, None
            
        except json.JSONDecodeError as e:
            logger.warning("parse_llm_response: JSON decode failed: %s", e)
            return raw_response, None
    
    # 尝试直接解析整个响应为 JSON（有些模型不加代码块）
    try:
        data = json.loads(raw_response)
        if isinstance(data, dict) and "answer" in data:
            answer = data["answer"]
            tool_data = data.get("tool_decision", {})
            if tool_data:
                decision = ToolDecision(
                    action=tool_data.get("action", "none"),
                    destination=tool_data.get("destination"),
                    path=tool_data.get("path"),
                    reason=tool_data.get("reason"),
                    pending_content=tool_data.get("pending_content"),
                )
                return answer, decision
            return answer, None
    except json.JSONDecodeError:
        pass
    
    # 解析失败，返回原始响应
    logger.info("parse_llm_response: no structured output detected, using raw response")
    return raw_response, None

