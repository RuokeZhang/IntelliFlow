from typing import List
from datetime import datetime

from app.db import session_scope
from app.models import MemorySummary
from app.schemas import MemorySummaryItem
from app.services.embedding import embed_texts


def save_summary(session_id: str, content: str) -> MemorySummaryItem:
    embedding = embed_texts([content])[0]
    with session_scope() as db:
        summary = MemorySummary(session_id=session_id, content=content, embedding=embedding)
        db.add(summary)
        db.flush()
        db.refresh(summary)
        return MemorySummaryItem(
            id=str(summary.id),
            content=summary.content,
            created_at=summary.created_at.isoformat(),
        )


def search_summaries(query: str, top_k: int = 3) -> List[MemorySummaryItem]:
    embed = embed_texts([query])[0]
    results = []
    with session_scope() as db:
        rows = (
            db.query(MemorySummary)
            .order_by(MemorySummary.embedding.l2_distance(embed))
            .limit(top_k)
            .all()
        )
        for row in rows:
            results.append(
                MemorySummaryItem(
                    id=str(row.id),
                    content=row.content,
                    created_at=row.created_at.isoformat(),
                )
            )
    return results


def list_summaries(session_id: str, limit: int = 5) -> List[MemorySummaryItem]:
    results = []
    with session_scope() as db:
        rows = (
            db.query(MemorySummary)
            .filter(MemorySummary.session_id == session_id)
            .order_by(MemorySummary.created_at.desc())
            .limit(limit)
            .all()
        )
        for row in rows:
            results.append(
                MemorySummaryItem(
                    id=str(row.id),
                    content=row.content,
                    created_at=row.created_at.isoformat(),
                )
            )
    return results

