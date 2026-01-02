import logging
from typing import Generator, Optional

from app.config import get_settings
from app.memory.session import SessionMemory
from app.memory.summary import save_summary
from app.rag.advisors import build_system_prompt, gather_context
from app.schemas import AgentRequest, AgentResult, PublishConfig
from app.services import llm
from app.agents.tools.github_publisher import GitHubPublisher
from app.agents.tools.local_file import LocalFileTool

logger = logging.getLogger(__name__)
settings = get_settings()


class AgentOrchestrator:
    def __init__(self):
        self.memory = SessionMemory()
        self.local_tool = LocalFileTool()
        try:
            self.github_tool = GitHubPublisher()
        except Exception as exc:
            logger.warning("GitHub 未配置，将跳过发布功能: %s", exc)
            self.github_tool = None

    def _build_messages(self, req: AgentRequest, contexts: list[str]) -> list[dict]:
        messages = build_system_prompt(req.prompt, contexts)
        history = self.memory.get_recent(req.session_id)
        for item in history:
            if item["type"] == "message":
                messages.append({"role": item["role"], "content": item["content"]})
            elif item["type"] == "tool":
                messages.append({"role": "system", "content": f"工具({item['tool_name']}): {item['content']}"})
        messages.append({"role": "user", "content": req.prompt})
        return messages

    def run(self, req: AgentRequest) -> AgentResult:
        contexts = gather_context(req.prompt, top_k=req.top_k)
        messages = self._build_messages(req, contexts)

        answer = llm.chat(messages)
        self.memory.add_message(req.session_id, "user", req.prompt)
        self.memory.add_message(req.session_id, "assistant", answer)

        publish_path = None
        publish_url = None
        if req.publish:
            publish_path, publish_url = self._handle_publish(answer, req.publish)

        if len(self.memory.get_recent(req.session_id)) > settings.summary_trigger_messages:
            try:
                summary = llm.chat(
                    [
                        {"role": "system", "content": "总结以下对话的关键信息供后续检索"},
                        {
                            "role": "user",
                            "content": "\n".join([m["content"] for m in self.memory.get_recent(req.session_id)]),
                        },
                    ]
                )
                save_summary(req.session_id, summary)
                self.memory.add_tool_event(req.session_id, "summary", "已生成摘要")
            except Exception as exc:
                logger.warning("摘要生成失败: %s", exc)

        return AgentResult(session_id=req.session_id, output=answer, publish_path=publish_path, publish_url=publish_url)

    def run_stream(self, req: AgentRequest) -> Generator[str, None, None]:
        contexts = gather_context(req.prompt, top_k=req.top_k)
        yield f"event: context\ndata: {len(contexts)}\n\n"

        messages = self._build_messages(req, contexts)
        collected = []
        for chunk in llm.chat_stream(messages):
            collected.append(chunk)
            yield f"event: llm\ndata: {chunk}\n\n"

        answer = "".join(collected)
        self.memory.add_message(req.session_id, "user", req.prompt)
        self.memory.add_message(req.session_id, "assistant", answer)

        if req.publish:
            try:
                path, url = self._handle_publish(answer, req.publish)
                yield f"event: publish\ndata: {path}\n\n"
                if url:
                    yield f"event: publish_url\ndata: {url}\n\n"
            except Exception as exc:
                yield f"event: error\ndata: 发布失败 {exc}\n\n"

        yield f"event: done\ndata: complete\n\n"

    def _handle_publish(self, content: str, publish: PublishConfig) -> tuple[Optional[str], Optional[str]]:
        if publish.mode == "github":
            if not self.github_tool:
                raise RuntimeError("GitHub 未配置")
            path = publish.path or f"{settings.github_base_path}/output.md"
            resp = self.github_tool.publish_markdown(path, content)
            url = resp.get("content", {}).get("html_url")
            self.memory.add_tool_event(publish.path or "github", "github_publisher", f"published to {path}")
            return path, url
        elif publish.mode == "local":
            path = publish.path or "output.md"
            target = self.local_tool.write(path, content)
            self.memory.add_tool_event(publish.path or "local", "local_file", f"write {target}")
            return str(target), None
        else:
            raise ValueError(f"未知发布模式: {publish.mode}")

