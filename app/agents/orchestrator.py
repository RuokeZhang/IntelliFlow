import logging
from typing import Generator, Optional

from app.config import get_settings
from app.memory.session import SessionMemory
from app.memory.summary import save_summary
from app.rag.advisors import build_system_prompt, gather_context, parse_llm_response
from app.schemas import AgentRequest, AgentResult, PublishConfig, ToolDecision
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

    def _build_messages(
        self, 
        req: AgentRequest, 
        contexts: list[str],
        enable_tool_decision: bool = True,
    ) -> list[dict]:
        """
        构建发送给 LLM 的消息列表。
        
        Args:
            req: 用户请求
            contexts: RAG 检索的上下文
            enable_tool_decision: 是否启用工具调用决策（结构化输出）
        """
        # 如果用户显式传了 publish 配置，则不需要 LLM 决策
        messages = build_system_prompt(
            req.prompt, 
            contexts, 
            enable_tool_decision=enable_tool_decision and req.publish is None
        )
        history = self.memory.get_recent(req.session_id)
        for item in history:
            if item["type"] == "message":
                messages.append({"role": item["role"], "content": item["content"]})
            elif item["type"] == "tool":
                messages.append({"role": "system", "content": f"工具({item['tool_name']}): {item['content']}"})
        messages.append({"role": "user", "content": req.prompt})
        return messages

    def run(self, req: AgentRequest) -> AgentResult:
        """
        执行 Agent 编排流程（同步模式）。
        
        流程：
        1. 收集 RAG 上下文
        2. 构建消息并调用 LLM
        3. 解析 LLM 结构化输出，提取 answer 和 tool_decision
        4. 根据决策执行工具调用（发布）
        5. 记录消息到 session memory
        6. 触发摘要生成（如果超过阈值）
        """
        contexts = gather_context(req.prompt, top_k=req.top_k)
        
        # 如果用户显式传了 publish，不启用 LLM 决策
        enable_tool_decision = req.publish is None
        messages = self._build_messages(req, contexts, enable_tool_decision=enable_tool_decision)

        raw_response = llm.chat(messages)
        logger.info("LLM raw response length: %d", len(raw_response))
        
        # 解析结构化输出
        if enable_tool_decision:
            answer, tool_decision = parse_llm_response(raw_response)
        else:
            answer = raw_response
            tool_decision = None
        
        # 记录消息到 session memory（存储给用户看的 answer，而非原始响应）
        self.memory.add_message(req.session_id, "user", req.prompt)
        self.memory.add_message(req.session_id, "assistant", answer)

        publish_path = None
        publish_url = None
        
        # 优先级：显式 publish 配置 > LLM 决策
        if req.publish:
            # 用户显式指定了发布配置
            logger.info("Using explicit publish config: mode=%s, path=%s", req.publish.mode, req.publish.path)
            publish_path, publish_url = self._handle_publish_config(answer, req.publish)
        elif tool_decision:
            # 根据 LLM 决策执行
            publish_path, publish_url = self._handle_tool_decision(
                answer, tool_decision, req.session_id
            )

        # 摘要生成
        if len(self.memory.get_recent(req.session_id)) > settings.summary_trigger_messages:
            self._generate_summary(req.session_id)

        return AgentResult(
            session_id=req.session_id, 
            output=answer, 
            publish_path=publish_path, 
            publish_url=publish_url
        )

    def run_stream(self, req: AgentRequest) -> Generator[str, None, None]:
        """
        执行 Agent 编排流程（SSE 流模式）。
        
        注意：流模式下无法实时解析结构化输出，因此：
        - 如果用户显式传了 publish，则按配置执行
        - 否则不执行工具调用（用户需要在下一轮明确指定）
        """
        contexts = gather_context(req.prompt, top_k=req.top_k)
        yield f"event: context\ndata: {len(contexts)}\n\n"

        # 流模式下禁用结构化输出（无法实时解析）
        messages = self._build_messages(req, contexts, enable_tool_decision=False)
        collected = []
        for chunk in llm.chat_stream(messages):
            collected.append(chunk)
            yield f"event: llm\ndata: {chunk}\n\n"

        answer = "".join(collected)
        self.memory.add_message(req.session_id, "user", req.prompt)
        self.memory.add_message(req.session_id, "assistant", answer)

        if req.publish:
            try:
                path, url = self._handle_publish_config(answer, req.publish)
                yield f"event: publish\ndata: {path}\n\n"
                if url:
                    yield f"event: publish_url\ndata: {url}\n\n"
            except Exception as exc:
                yield f"event: error\ndata: 发布失败 {exc}\n\n"

        yield f"event: done\ndata: complete\n\n"

    def _get_pending_content(self, session_id: str) -> Optional[str]:
        """
        从 session memory 中获取待发布的内容（pending_publish 事件）。
        """
        history = self.memory.get_recent(session_id)
        for item in reversed(history):  # 从最近的往前找
            if item.get("type") == "tool" and item.get("tool_name") == "pending_publish":
                return item.get("content")
        return None

    def _clear_pending_content(self, session_id: str) -> None:
        """
        清除 pending_publish 事件（发布完成后）。
        注意：当前实现是在 memory 中添加一个 "cleared" 标记，
        因为 Redis List 不方便删除特定元素。
        """
        self.memory.add_tool_event(session_id, "pending_publish_cleared", "已发布")

    def _handle_tool_decision(
        self, 
        content: str, 
        decision: ToolDecision,
        session_id: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        根据 LLM 的工具调用决策执行操作。
        
        Args:
            content: 当前回答内容（可能用于发布）
            decision: LLM 的决策
            session_id: 会话 ID（用于记录工具事件）
            
        Returns:
            (publish_path, publish_url)
        """
        logger.info(
            "Handling tool decision: action=%s, destination=%s, path=%s, has_pending=%s",
            decision.action, decision.destination, decision.path,
            bool(decision.pending_content)
        )
        
        if decision.action == "none":
            # 不需要发布
            return None, None
        
        if decision.action == "ask_destination":
            # LLM 已在 answer 中追问用户，暂存待发布内容
            pending = decision.pending_content or content
            if pending:
                self.memory.add_tool_event(session_id, "pending_publish", pending)
                logger.info("Stored pending content for later publish, length=%d", len(pending))
            return None, None
        
        if decision.action == "publish":
            # 确定要发布的内容：优先使用 pending_content，其次是 session 中暂存的，最后是当前 answer
            publish_content = decision.pending_content
            if not publish_content:
                publish_content = self._get_pending_content(session_id)
            if not publish_content:
                publish_content = content
            
            if decision.destination == "local":
                path = decision.path or "output.md"
                target = self.local_tool.write(path, publish_content)
                self.memory.add_tool_event(session_id, "local_file", f"write {target}")
                self._clear_pending_content(session_id)
                logger.info("Published to local: %s", target)
                return str(target), None
            
            elif decision.destination == "github":
                if not self.github_tool:
                    logger.warning("GitHub 未配置，无法发布")
                    return None, None
                path = decision.path or f"{settings.github_base_path}/output.md"
                resp = self.github_tool.publish_markdown(path, publish_content)
                url = resp.get("content", {}).get("html_url")
                self.memory.add_tool_event(session_id, "github_publisher", f"published to {path}")
                self._clear_pending_content(session_id)
                logger.info("Published to GitHub: %s -> %s", path, url)
                return path, url
            
            else:
                logger.warning("Unknown destination: %s", decision.destination)
                return None, None
        
        return None, None

    def _handle_publish_config(
        self, 
        content: str, 
        publish: PublishConfig
    ) -> tuple[Optional[str], Optional[str]]:
        """
        根据用户显式传入的 PublishConfig 执行发布。
        """
        if publish.mode == "github":
            if not self.github_tool:
                raise RuntimeError("GitHub 未配置")
            path = publish.path or f"{settings.github_base_path}/output.md"
            resp = self.github_tool.publish_markdown(path, content)
            url = resp.get("content", {}).get("html_url")
            return path, url
        elif publish.mode == "local":
            path = publish.path or "output.md"
            target = self.local_tool.write(path, content)
            return str(target), None
        else:
            raise ValueError(f"未知发布模式: {publish.mode}")

    def _generate_summary(self, session_id: str) -> None:
        """
        生成会话摘要并持久化。
        """
        try:
            summary = llm.chat(
                [
                    {"role": "system", "content": "总结以下对话的关键信息供后续检索"},
                    {
                        "role": "user",
                        "content": "\n".join([m["content"] for m in self.memory.get_recent(session_id)]),
                    },
                ]
            )
            save_summary(session_id, summary)
            self.memory.add_tool_event(session_id, "summary", "已生成摘要")
            logger.info("Summary generated for session: %s", session_id)
        except Exception as exc:
            logger.warning("摘要生成失败: %s", exc)
