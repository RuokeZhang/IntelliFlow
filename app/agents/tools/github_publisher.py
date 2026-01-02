import base64
from typing import Optional

import httpx

from app.config import get_settings

settings = get_settings()


class GitHubPublisher:
    def __init__(self):
        if not settings.github_token or not settings.github_repo:
            raise RuntimeError("GitHub 配置缺失（github_token 或 github_repo）")
        self.repo = settings.github_repo
        self.branch = settings.github_branch
        self.client = httpx.Client(
            base_url="https://api.github.com",
            headers={"Authorization": f"Bearer {settings.github_token}", "Accept": "application/vnd.github+json"},
            timeout=20.0,
        )

    def publish_markdown(
        self, path: str, content: str, message: str = "chore: publish from IntelliFlow"
    ) -> dict:
        url = f"/repos/{self.repo}/contents/{path}"
        sha = self._get_existing_sha(url)
        payload = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
            "branch": self.branch,
        }
        if sha:
            payload["sha"] = sha
        resp = self.client.put(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def _get_existing_sha(self, url: str) -> Optional[str]:
        resp = self.client.get(url, params={"ref": self.branch})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data.get("sha")

