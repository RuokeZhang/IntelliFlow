from typing import List
import logging

from app.config import get_settings
from app.db import session_scope
from app.models import DocumentChunk, Document
from app.schemas import RetrievedChunk
from app.services.embedding import embed_texts
from app.services.reranker import reranker

logger = logging.getLogger(__name__)
settings = get_settings()


def retrieve(query: str, top_k: int | None = None) -> List[RetrievedChunk]:
    """
    两阶段检索：
    1. 粗排 (Coarse-grained): 向量搜索召回 top_n 个候选
    2. 精排 (Fine-grained): Reranker 重新打分并截取 top_k
    """
    embed = embed_texts([query])[0]
    final_limit = top_k or settings.top_k
    # 粗排召回数量需要大于最终返回数量
    coarse_limit = settings.rerank_top_n if settings.use_rerank else final_limit

    coarse_results: List[RetrievedChunk] = []
    
    with session_scope() as db:
        distance = DocumentChunk.embedding.l2_distance(embed).label("distance")
        
        rows = (
            db.query(DocumentChunk, Document, distance)
            .join(Document, DocumentChunk.document_id == Document.id)
            .order_by(distance)
            .limit(coarse_limit)
            .all()
        )

        for chunk, doc, dist in rows:
            coarse_results.append(
                RetrievedChunk(
                    document_id=str(doc.id),
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=float(dist),
                )
            )

    # 判定是否启用 Rerank
    if settings.use_rerank and reranker.client and len(coarse_results) > 1:
        logger.info(f"Performing Rerank for query: {query}")
        doc_contents = [res.content for res in coarse_results]
        reranked = reranker.rerank(query, doc_contents, top_n=final_limit)
        
        final_results = []
        for item in reranked:
            original = coarse_results[item["index"]]
            original.score = item["relevance_score"] # 替换为精排分数
            final_results.append(original)
        return final_results
    
    # 未启用或失败降级：直接返回粗排结果的前 top_k
    return coarse_results[:final_limit]

