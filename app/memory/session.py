"""
生产级 Session Memory 实现

支持两种存储后端：
1. Redis（推荐生产环境）：持久化、支持 TTL 自动过期、可水平扩展
2. 内存（开发/Demo）：简单快速，但重启丢失

使用方式：
- 配置 REDIS_URL 环境变量后自动使用 Redis
- 未配置时降级为内存模式
"""
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseSessionMemory(ABC):
    """Session Memory 抽象基类"""

    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加一条对话消息"""
        pass

    @abstractmethod
    def add_tool_event(self, session_id: str, tool_name: str, content: str) -> None:
        """添加一条工具调用事件"""
        pass

    @abstractmethod
    def get_recent(self, session_id: str) -> List[dict]:
        """获取最近的对话历史"""
        pass

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """清空指定 session 的历史"""
        pass

    @abstractmethod
    def touch(self, session_id: str) -> None:
        """刷新 session 的过期时间"""
        pass

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """获取或创建 session_id"""
        if session_id:
            self.touch(session_id)
            return session_id
        return str(uuid4())


class InMemorySessionMemory(BaseSessionMemory):
    """
    内存实现（开发/Demo 模式）
    
    特点：
    - 简单快速，无外部依赖
    - 支持 TTL 过期（惰性检查）
    - 重启后数据丢失
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ):
        self.window_size = window_size or settings.session_window_size
        self.ttl = ttl_seconds or settings.session_ttl_seconds
        self._store: Dict[str, List[dict]] = defaultdict(list)
        self._last_active: Dict[str, float] = {}
        self._lock = Lock()

    def _is_expired(self, session_id: str) -> bool:
        """检查 session 是否过期"""
        last = self._last_active.get(session_id, 0)
        return (time.time() - last) > self.ttl

    def _cleanup_expired(self, session_id: str) -> None:
        """清理过期的 session"""
        if self._is_expired(session_id):
            self._store.pop(session_id, None)
            self._last_active.pop(session_id, None)

    def touch(self, session_id: str) -> None:
        with self._lock:
            self._last_active[session_id] = time.time()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._cleanup_expired(session_id)
            self._store[session_id].append({
                "type": "message",
                "role": role,
                "content": content,
                "timestamp": time.time(),
            })
            self._last_active[session_id] = time.time()
            self._trim(session_id)

    def add_tool_event(self, session_id: str, tool_name: str, content: str) -> None:
        with self._lock:
            self._cleanup_expired(session_id)
            self._store[session_id].append({
                "type": "tool",
                "tool_name": tool_name,
                "content": content,
                "timestamp": time.time(),
            })
            self._last_active[session_id] = time.time()
            self._trim(session_id)

    def get_recent(self, session_id: str) -> List[dict]:
        with self._lock:
            self._cleanup_expired(session_id)
            return list(self._store.get(session_id, []))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)
            self._last_active.pop(session_id, None)

    def _trim(self, session_id: str) -> None:
        """滑动窗口裁剪"""
        if len(self._store[session_id]) > self.window_size:
            overflow = len(self._store[session_id]) - self.window_size
            self._store[session_id] = self._store[session_id][overflow:]


class RedisSessionMemory(BaseSessionMemory):
    """
    Redis 实现（生产环境推荐）
    
    特点：
    - 持久化存储，服务重启不丢失
    - 原生 TTL 支持，自动过期
    - 支持分布式部署、水平扩展
    - 高性能，适合高并发场景
    """

    def __init__(
        self,
        redis_url: str,
        window_size: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ):
        try:
            import redis
        except ImportError:
            raise ImportError("请安装 redis: pip install redis")

        self.client = redis.from_url(redis_url, decode_responses=True)
        self.window_size = window_size or settings.session_window_size
        self.ttl = ttl_seconds or settings.session_ttl_seconds
        self._key_prefix = "intelliflow:session:"
        self._script = self.client.register_script(
            """
            redis.call('RPUSH', KEYS[1], ARGV[1])
            redis.call('LTRIM', KEYS[1], ARGV[2], -1)
            redis.call('EXPIRE', KEYS[1], ARGV[3])
            """
        )

        # 测试连接
        try:
            self.client.ping()
            logger.info("Redis 连接成功")
        except Exception as e:
            logger.error(f"Redis 连接失败: {e}")
            raise

    def _get_key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    def touch(self, session_id: str) -> None:
        """刷新 TTL"""
        key = self._get_key(session_id)
        self.client.expire(key, self.ttl)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        key = self._get_key(session_id)
        msg = json.dumps({
            "type": "message",
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }, ensure_ascii=False)
        
        self._script(keys=[key], args=[msg, -self.window_size, self.ttl])

    def add_tool_event(self, session_id: str, tool_name: str, content: str) -> None:
        key = self._get_key(session_id)
        msg = json.dumps({
            "type": "tool",
            "tool_name": tool_name,
            "content": content,
            "timestamp": time.time(),
        }, ensure_ascii=False)
        
        self._script(keys=[key], args=[msg, -self.window_size, self.ttl])

    def get_recent(self, session_id: str) -> List[dict]:
        key = self._get_key(session_id)
        items = self.client.lrange(key, 0, -1)
        return [json.loads(item) for item in items]

    def clear(self, session_id: str) -> None:
        key = self._get_key(session_id)
        self.client.delete(key)


class SessionMemory(BaseSessionMemory):
    """
    Session Memory 工厂类
    
    自动选择后端：
    - 配置了 REDIS_URL → 使用 Redis
    - 未配置 → 使用内存（会打印警告）
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ):
        if settings.redis_url:
            logger.info("使用 Redis 作为 Session 存储后端")
            self._backend = RedisSessionMemory(
                redis_url=settings.redis_url,
                window_size=window_size,
                ttl_seconds=ttl_seconds,
            )
        else:
            logger.warning(
                "未配置 REDIS_URL，使用内存存储。"
                "生产环境建议配置 Redis 以支持持久化和分布式部署。"
            )
            self._backend = InMemorySessionMemory(
                window_size=window_size,
                ttl_seconds=ttl_seconds,
            )

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._backend.add_message(session_id, role, content)

    def add_tool_event(self, session_id: str, tool_name: str, content: str) -> None:
        self._backend.add_tool_event(session_id, tool_name, content)

    def get_recent(self, session_id: str) -> List[dict]:
        return self._backend.get_recent(session_id)

    def clear(self, session_id: str) -> None:
        self._backend.clear(session_id)

    def touch(self, session_id: str) -> None:
        self._backend.touch(session_id)

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        return self._backend.get_or_create_session(session_id)
