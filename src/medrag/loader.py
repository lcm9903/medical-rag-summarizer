"""의료 문서 로딩과 섹션 단위 청킹.

문서는 YAML 스타일 front matter(메타데이터) + `## 섹션` 구조의 마크다운입니다.
의무기록은 섹션(주호소, 진단명, 퇴원 약물 등)마다 의미가 달라서,
글자 수로만 자르기보다 섹션 경계를 먼저 지키고 긴 섹션만 추가로 나눕니다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """LangChain의 Document와 같은 형태(page_content, metadata)를 갖는 경량 클래스."""

    page_content: str
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.metadata.get('doc_id', '?')}#{self.metadata.get('section', '?')}"

    def to_langchain(self):
        from langchain_core.documents import Document as LCDocument

        return LCDocument(page_content=self.page_content, metadata=dict(self.metadata))


_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def parse_note(text: str) -> tuple[dict, list[tuple[str, str]]]:
    """문서 텍스트를 (메타데이터, [(섹션명, 본문), ...])으로 분리."""
    meta: dict = {}
    body = text
    m = _FRONT_MATTER.match(text)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
        body = text[m.end():]

    sections: list[tuple[str, str]] = []
    matches = list(_SECTION.finditer(body))
    for i, sm in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[sm.end():end].strip()
        if content:
            sections.append((sm.group(1).strip(), content))
    return meta, sections


def _split_long(text: str, chunk_size: int, overlap: int) -> list[str]:
    """긴 섹션을 문장/줄 경계 기준으로 나누고, 앞 조각의 끝부분을 overlap만큼 겹칩니다."""
    if len(text) <= chunk_size:
        return [text]
    units = [u for u in re.split(r"(?<=[.!?。])\s+|\n", text) if u.strip()]
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) > chunk_size and current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {unit}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def load_notes(notes_dir: Path | str, chunk_size: int = 400, chunk_overlap: int = 50) -> list[Document]:
    """폴더의 모든 .md 문서를 읽어 섹션 단위 Document 청크 목록으로 반환."""
    notes_dir = Path(notes_dir)
    docs: list[Document] = []
    for path in sorted(notes_dir.glob("*.md")):
        meta, sections = parse_note(path.read_text(encoding="utf-8"))
        meta.setdefault("doc_id", path.stem)
        meta["source"] = path.name
        for section, content in sections:
            for i, piece in enumerate(_split_long(content, chunk_size, chunk_overlap)):
                docs.append(Document(page_content=piece, metadata={**meta, "section": section, "chunk": i}))
    return docs
