from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

from app.config import get_settings

settings = get_settings()

client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def embed_texts(texts: List[str]) -> List[List[float]]:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY 未配置，无法生成向量")

    resp = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in resp.data]

