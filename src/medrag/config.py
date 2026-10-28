"""환경 변수 기반 설정."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    # LLM: dryrun(오프라인 규칙 기반) | openai | ollama
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "dryrun"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))

    # 검색기: tfidf(오프라인) | faiss(임베딩 벡터 검색)
    retriever: str = field(default_factory=lambda: os.getenv("RETRIEVER", "tfidf"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
    )
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))

    # 청킹
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "400")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))

    # 검증 실패 시 재요약 최대 횟수
    max_revisions: int = field(default_factory=lambda: int(os.getenv("MAX_REVISIONS", "1")))

    notes_dir: Path = field(
        default_factory=lambda: Path(os.getenv("NOTES_DIR", PROJECT_ROOT / "data" / "notes"))
    )
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("DB_PATH", PROJECT_ROOT / "outputs" / "summaries.db"))
    )


def get_settings() -> Settings:
    return Settings()
