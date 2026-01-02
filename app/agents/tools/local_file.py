import os
from pathlib import Path

from app.config import get_settings

settings = get_settings()


class LocalFileTool:
    def __init__(self):
        self.base = Path(settings.local_workspace).resolve()
        self.base.mkdir(parents=True, exist_ok=True)

    def write(self, relative_path: str, content: str) -> Path:
        target = (self.base / relative_path).resolve()
        if not str(target).startswith(str(self.base)):
            raise ValueError("路径越界")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def read(self, relative_path: str) -> str:
        target = (self.base / relative_path).resolve()
        if not target.exists():
            raise FileNotFoundError(relative_path)
        return target.read_text(encoding="utf-8")

