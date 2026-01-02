from typing import Iterable, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

from app.config import get_settings

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def chat(messages: List[dict], temperature: float = 0.2, tools: Optional[list] = None) -> str:
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=temperature,
        tools=tools,
    )
    return resp.choices[0].message.content or ""


def chat_stream(messages: List[dict], temperature: float = 0.2) -> Iterable[str]:
    stream = client.chat.completions.create(
        model=settings.openai_model, messages=messages, temperature=temperature, stream=True
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

