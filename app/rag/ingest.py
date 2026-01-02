import re
import tiktoken
from dataclasses import dataclass
from typing import List, Tuple

from bs4 import BeautifulSoup
from tika import parser

from app.config import get_settings
from app.db import session_scope
from app.models import Document, DocumentChunk
from app.services.embedding import embed_texts

settings = get_settings()


@dataclass
class Section:
    """表示一个结构化章节"""
    title_path: str  # 层次路径，如 "分块策略 > 语义分块"
    content: str     # 章节正文


def load_text(path: str) -> str:
    """加载文件为纯文本（用于非结构化文本文件）"""
    parsed = parser.from_file(path)
    content = parsed.get("content") or ""
    return content.strip()


def load_xhtml_from_file(path: str) -> str:
    """
    使用 Tika 的 XHTML 输出模式加载文件，保留结构信息。
    支持 PDF、Word、PPT 等格式，返回带有 <h1>/<h2>/<p>/<table> 等标签的 XHTML。
    """
    parsed = parser.from_file(path, xmlContent=True)
    content = parsed.get("content") or ""
    return content.strip()


def _get_encoding():
    return tiktoken.encoding_for_model(settings.openai_model)


def _count_tokens(text: str) -> int:
    return len(_get_encoding().encode(text))


def chunk_text(text: str, size: int, overlap: int, prefix: str = "") -> List[str]:
    """
    使用 tiktoken 进行 Token 级分块，支持多语言。
    prefix: 可选的章节路径前缀，会添加到每个 chunk 开头。
    """
    encoding = _get_encoding()
    
    # 计算前缀占用的 tokens
    prefix_str = f"[{prefix}] " if prefix else ""
    prefix_tokens = len(encoding.encode(prefix_str))
    effective_size = size - prefix_tokens
    
    tokens = encoding.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + effective_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(prefix_str + chunk_text)

        if end == len(tokens):
            break
        start = end - overlap

    return chunks


def parse_html_sections(html: str) -> List[Section]:
    """
    解析 HTML 结构，提取章节层次。
    返回 Section 列表，每个 Section 包含标题路径和正文。
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # 移除非正文元素
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    
    sections: List[Section] = []
    title_stack: List[str] = []  # 当前标题层次栈
    current_content: List[str] = []
    current_level = 0
    
    # 遍历所有元素
    for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "pre", "ul", "ol", "table", "blockquote"]):
        tag_name = elem.name
        
        if tag_name in ["h1", "h2", "h3", "h4"]:
            # 保存之前的章节
            if current_content:
                title_path = " > ".join(title_stack) if title_stack else "正文"
                content = "\n".join(current_content).strip()
                if content:
                    sections.append(Section(title_path=title_path, content=content))
                current_content = []
            
            # 更新标题层次
            level = int(tag_name[1])
            title_text = elem.get_text(strip=True)
            
            if level > current_level:
                title_stack.append(title_text)
            elif level == current_level:
                if title_stack:
                    title_stack[-1] = title_text
                else:
                    title_stack.append(title_text)
            else:
                # 回退到对应层级
                while len(title_stack) > level - 1:
                    title_stack.pop()
                title_stack.append(title_text)
            
            current_level = level
        else:
            # 正文内容
            text = elem.get_text(separator=" ", strip=True)
            if text:
                current_content.append(text)
    
    # 保存最后一个章节
    if current_content:
        title_path = " > ".join(title_stack) if title_stack else "正文"
        content = "\n".join(current_content).strip()
        if content:
            sections.append(Section(title_path=title_path, content=content))
    
    return sections


def chunk_sections(
    sections: List[Section],
    size: int,
    overlap: int,
) -> List[str]:
    """
    对章节列表进行智能分块：
    - 短章节直接作为一个 chunk（带标题前缀）
    - 长章节用 Token 滑窗二次切分（每个子 chunk 都带标题前缀）
    """
    chunks: List[str] = []
    
    for section in sections:
        prefix = section.title_path
        prefix_str = f"[{prefix}] "
        full_text = prefix_str + section.content
        
        token_count = _count_tokens(full_text)
        
        if token_count <= size:
            # 短章节，直接作为一个 chunk
            chunks.append(full_text)
        else:
            # 长章节，二次切分
            sub_chunks = chunk_text(section.content, size, overlap, prefix=prefix)
            chunks.extend(sub_chunks)
    
    return chunks


def ingest_html(
    html: str,
    source: str = "url",
    uri: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Tuple[str, int]:
    """
    结构感知的 HTML 入库：解析章节结构，智能分块，带标题传播。
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    
    sections = parse_html_sections(html)
    
    if not sections:
        raise ValueError("未能从 HTML 中提取到有效章节")
    
    chunks = chunk_sections(sections, size, overlap)
    
    if not chunks:
        raise ValueError("分块结果为空")
    
    embeddings = embed_texts(chunks)
    
    with session_scope() as db:
        doc = Document(source=source, uri=uri)
        db.add(doc)
        db.flush()
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            db.add(
                DocumentChunk(
                    document_id=doc.id,
                    chunk_index=idx,
                    content=chunk,
                    embedding=emb,
                )
            )
        return str(doc.id), len(chunks)


def ingest_file(
    path: str,
    source: str = "local",
    uri: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Tuple[str, int]:
    """
    结构感知的文件入库：使用 Tika XHTML 输出，复用 HTML 解析逻辑。
    支持 PDF、Word、PPT 等格式。
    """
    xhtml = load_xhtml_from_file(path)
    if not xhtml:
        raise ValueError(f"文件内容为空: {path}")

    # 复用 HTML 结构感知分块逻辑
    return ingest_html(
        xhtml,
        source=source,
        uri=uri or path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def ingest_text(
    text: str,
    source: str = "upload",
    uri: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Tuple[str, int]:
    if not text:
        raise ValueError("文本内容为空")

    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    chunks = chunk_text(text, size, overlap)
    embeddings = embed_texts(chunks)

    with session_scope() as db:
        doc = Document(source=source, uri=uri)
        db.add(doc)
        db.flush()
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            db.add(
                DocumentChunk(
                    document_id=doc.id,
                    chunk_index=idx,
                    content=chunk,
                    embedding=emb,
                )
            )
        return str(doc.id), len(chunks)

