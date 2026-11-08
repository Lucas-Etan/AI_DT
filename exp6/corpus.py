import re
from pathlib import Path

from schemas import Chunk


def read_documents(docs_dir: Path) -> list[tuple[Path, str]]:
    """按文件名排序读取 Markdown/文本，保证每次索引顺序一致。"""
    paths = sorted([*docs_dir.glob("*.md"), *docs_dir.glob("*.txt")])
    documents = []
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append((path, text))
    if not documents:
        raise ValueError(f"知识库为空：{docs_dir}")
    return documents


def split_document(path: Path, text: str, size: int, overlap: int) -> list[Chunk]:
    """先按空行分段；过长段落再用带重叠的字符窗口切分。"""
    if size <= 0:
        raise ValueError("窗口大小必须为正整数")
    if not 0 <= overlap < size:
        raise ValueError("overlap 必须满足 0 <= overlap < size")

    title = ""
    pending_heading = ""
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if block.startswith("# ") and not block.startswith("## "):
            title = block[2:].strip()
        elif block.startswith("## ") and "\n" not in block:
            pending_heading = block
        else:
            if pending_heading:
                paragraphs.append(f"{pending_heading}\n{block}")
                pending_heading = ""
            else:
                paragraphs.append(block)

    chunks: list[Chunk] = []
    for paragraph_no, paragraph in enumerate(paragraphs, start=1):
        starts = range(0, len(paragraph), size - overlap)
        for window_no, start in enumerate(starts, start=1):
            piece = paragraph[start:start + size].strip()
            if not piece:
                continue
            end = min(start + size, len(paragraph))
            chunk_id = f"{path.stem}-p{paragraph_no:02d}-w{window_no:02d}"
            chunks.append(Chunk(chunk_id=chunk_id, source=path.name,
                                title=title, paragraph=paragraph_no,
                                char_start=start, char_end=end, text=piece))
            if start + size >= len(paragraph):
                break
    return chunks


def load_chunks(docs_dir: Path, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, text in read_documents(docs_dir):
        chunks.extend(split_document(path, text, size, overlap))
    return chunks
