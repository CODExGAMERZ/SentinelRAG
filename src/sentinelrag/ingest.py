from __future__ import annotations

import csv
import hashlib
import html.parser
from datetime import UTC, datetime
from pathlib import Path

from .types import ChunkRecord

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".html", ".htm", ".pdf"}
SUPPORTED_TEXT_NAMES = {".env"}
MAX_READ_BYTES = 25 * 1024 * 1024
MAX_CHUNKS_PER_FILE = 256


class _TextExtractor(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def read_document(path: Path) -> str:
    _enforce_file_size(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"} or path.name.lower() in SUPPORTED_TEXT_NAMES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".csv":
        rows: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                rows.append(" | ".join(cell.strip() for cell in row))
        return "\n".join(rows)
    if suffix in {".html", ".htm"}:
        parser = _TextExtractor()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        return "\n".join(parser.parts)
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:
            raise RuntimeError("PDF ingestion requires optional dependency: pip install sentinelrag[pdf]") from exc
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if is_supported_file(path) else []
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and is_supported_file(file) and ".sentinelrag" not in file.parts
    )


def is_supported_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS or path.name.lower() in SUPPORTED_TEXT_NAMES


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 160) -> list[str]:
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks[:MAX_CHUNKS_PER_FILE] if chunk]


def document_id(path: Path, text: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.resolve()).encode("utf-8", errors="replace"))
    digest.update(text.encode("utf-8", errors="replace"))
    return digest.hexdigest()[:16]


def make_chunks(path: Path) -> list[ChunkRecord]:
    text = read_document(path)
    doc_id = document_id(path, text)
    created_at = datetime.now(UTC).isoformat()
    chunks = chunk_text(text)
    records: list[ChunkRecord] = []
    for index, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}:{index:04d}"
        records.append(
            ChunkRecord(
                doc_id=doc_id,
                chunk_id=chunk_id,
                source_path=str(path.resolve()),
                text=chunk,
                metadata={"extension": path.suffix.lower(), "chunk_index": index},
                created_at=created_at,
            )
        )
    return records


def _enforce_file_size(path: Path) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"Could not read file size: {path}") from exc
    if size > MAX_READ_BYTES:
        raise RuntimeError(
            f"File exceeds safe ingest limit ({MAX_READ_BYTES // (1024 * 1024)} MB): {path}"
        )
