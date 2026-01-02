import tempfile
from pathlib import Path

import httpx
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.config import get_settings
from app.rag.ingest import ingest_file, ingest_text, ingest_html
from app.schemas import IngestResponse, UrlIngestRequest

router = APIRouter()
settings = get_settings()


@router.post("/upload", response_model=IngestResponse)
async def upload_file(
    file: UploadFile = File(...),
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    suffix = Path(file.filename or "").suffix.lower()
    # 对文本类型直接读取；其他类型落盘用 Tika 解析
    try:
        if suffix in {".txt", ".md", ".markdown", ".csv", ".json"}:
            text = (await file.read()).decode("utf-8", errors="ignore")
            doc_id, count = ingest_text(
                text,
                source="upload",
                uri=file.filename or "upload",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            # PDF、Word、PPT 等：使用 Tika XHTML 输出 + 结构感知分块
            with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp.flush()
                doc_id, count = ingest_file(
                    tmp.name,
                    source="upload",
                    uri=file.filename or tmp.name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"ingest failed: {exc}")
    return IngestResponse(document_id=doc_id, chunks=count)


@router.post("/url", response_model=IngestResponse)
async def ingest_url(body: UrlIngestRequest):
    """
    结构感知的网页导入：解析 HTML 章节结构，智能分块，带标题传播。
    """
    try:
        resp = httpx.get(body.url, timeout=20)
        resp.raise_for_status()
        html = resp.text
        
        # 使用结构感知分块
        doc_id, count = ingest_html(
            html,
            source=body.source or "url",
            uri=body.url,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
        )
        return IngestResponse(document_id=doc_id, chunks=count)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"url ingest failed: {exc}")

