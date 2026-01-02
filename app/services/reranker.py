import cohere
import logging
from typing import List, Dict, Any
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class Reranker:
    def __init__(self):
        if not settings.cohere_api_key:
            self.client = None
            logger.warning("COHERE_API_KEY 未配置，Rerank 功能将不可用")
        else:
            self.client = cohere.ClientV2(settings.cohere_api_key)

    def rerank(self, query: str, documents: List[str], top_n: int) -> List[Dict[str, Any]]:
        """
        使用 Cohere Rerank 对文档进行精排。
        返回格式: [{"index": int, "relevance_score": float}, ...]
        """
        if not self.client or not documents:
            return [{"index": i, "relevance_score": 0.0} for i in range(len(documents))][:top_n]

        try:
            response = self.client.rerank(
                model="rerank-multilingual-v3.0",
                query=query,
                documents=documents,
                top_n=top_n,
            )
            
            results = []
            for result in response.results:
                results.append({
                    "index": result.index,
                    "relevance_score": result.relevance_score
                })
            return results
        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            # 失败时降级，按原顺序返回
            return [{"index": i, "relevance_score": 0.0} for i in range(len(documents))][:top_n]

# 单例模式
reranker = Reranker()

