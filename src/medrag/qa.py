"""근거 인용을 포함한 의무기록 질의응답 (RAG)."""
from __future__ import annotations

from dataclasses import dataclass

from . import prompts
from .llm import LLM
from .retriever import Retriever


@dataclass
class Answer:
    answer: str
    sources: list[str]


class MedicalQA:
    def __init__(self, retriever: Retriever, llm: LLM, k: int = 4):
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def ask(self, question: str, patient_id: str | None = None) -> Answer:
        docs = self.retriever.invoke(question, patient_id=patient_id, k=self.k)
        if not docs:
            return Answer("기록에서 확인할 수 없습니다.", [])
        prompt = prompts.QA.format(
            question=question,
            context=prompts.format_context(docs),
            context_start=prompts.CONTEXT_START,
            context_end=prompts.CONTEXT_END,
        )
        return Answer(self.llm.invoke(prompt).strip(), [d.citation for d in docs])
