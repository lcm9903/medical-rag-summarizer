"""멀티 에이전트 구성 요소.

1. RetrievalAgent  — 요약에 필요한 섹션별 질의를 여러 개 던져 근거를 모읍니다(multi-query retrieval).
2. SummarizerAgent — 근거만으로 구조화된 퇴원요약(JSON)을 작성합니다.
3. VerifierAgent   — 요약의 약물·용량·수치·인용을 근거와 대조하고(규칙 기반),
                     선택적으로 LLM 판정자(LLM-as-a-judge)로 근거 없는 주장을 찾습니다.
검증에 실패하면 문제 목록을 피드백으로 넘겨 SummarizerAgent가 다시 작성합니다(graph.py).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import prompts
from .llm import LLM
from .loader import Document
from .retriever import Retriever
from .schema import DischargeSummary, parse_summary_json

SECTION_QUERIES = [
    "주호소 내원 증상",
    "진단명",
    "입원 경과 치료 시술",
    "퇴원 약물 용량 복용",
    "추적 관찰 계획 외래",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


class RetrievalAgent:
    def __init__(self, retriever: Retriever, per_query_k: int = 2):
        self.retriever = retriever
        self.per_query_k = per_query_k

    def run(self, patient_id: str) -> list[Document]:
        seen: set[tuple[str, int]] = set()
        evidence: list[Document] = []
        for query in SECTION_QUERIES:
            for doc in self.retriever.invoke(query, patient_id=patient_id, k=self.per_query_k):
                key = (doc.citation, doc.metadata.get("chunk", 0))
                if key not in seen:
                    seen.add(key)
                    evidence.append(doc)
        return evidence


class SummarizerAgent:
    def __init__(self, llm: LLM, max_parse_retries: int = 1):
        self.llm = llm
        self.max_parse_retries = max_parse_retries

    def run(self, patient_id: str, evidence: list[Document], issues: list[str] | None = None) -> DischargeSummary:
        feedback = ""
        if issues:
            feedback = prompts.REVISION_FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues))
        prompt = prompts.SUMMARIZER.format(
            patient_id=patient_id,
            feedback=feedback,
            context=prompts.format_context(evidence),
            context_start=prompts.CONTEXT_START,
            context_end=prompts.CONTEXT_END,
        )
        last_error: Exception | None = None
        for _ in range(self.max_parse_retries + 1):
            raw = self.llm.invoke(prompt)
            try:
                return parse_summary_json(raw, patient_id)
            except (ValueError, json.JSONDecodeError) as exc:  # JSON 형식이 깨지면 한 번 더 시도
                last_error = exc
        raise RuntimeError(f"요약 생성 실패: {last_error}")


@dataclass
class VerificationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


_CLAIM_NUMBER = re.compile(r"\d+(?:[.,]\d+)?\s*(?:점|mg/dL|mmHg|%|시간|일|주|개월|mg|L/분|도|단위)")


class VerifierAgent:
    def __init__(self, llm: LLM | None = None, use_llm_judge: bool = True):
        self.llm = llm
        self.use_llm_judge = use_llm_judge and llm is not None

    def rule_checks(self, summary: DischargeSummary, evidence: list[Document]) -> list[str]:
        context = _normalize(" ".join(d.page_content for d in evidence))
        issues: list[str] = []

        if not summary.chief_complaint.strip():
            issues.append("주호소가 비어 있습니다.")
        if not summary.diagnoses:
            issues.append("진단명이 비어 있습니다.")

        for med in summary.discharge_medications:
            # '아스피린(Aspirin)' → 괄호 앞 이름으로도 대조
            base = med.name.split("(")[0]
            if _normalize(med.name) not in context and _normalize(base) not in context:
                issues.append(f"근거에 없는 약물: {med.name}")
            elif med.dose and _normalize(med.dose) not in context:
                issues.append(f"근거와 다른 용량: {med.name} {med.dose}")

        for field_name in ("chief_complaint", "hospital_course", "follow_up"):
            for number in _CLAIM_NUMBER.findall(getattr(summary, field_name)):
                if _normalize(number) not in context:
                    issues.append(f"근거에 없는 수치({field_name}): {number}")

        valid_citations = {d.citation for d in evidence}
        for cite in summary.sources:
            if cite not in valid_citations:
                issues.append(f"존재하지 않는 인용: {cite}")
        return issues

    def llm_checks(self, summary: DischargeSummary, evidence: list[Document]) -> list[str]:
        prompt = prompts.JUDGE.format(
            summary=summary.model_dump_json(exclude={"sources"}),
            context=prompts.format_context(evidence),
            context_start=prompts.CONTEXT_START,
            context_end=prompts.CONTEXT_END,
        )
        raw = self.llm.invoke(prompt)
        try:
            start, end = raw.find("{"), raw.rfind("}")
            unsupported = json.loads(raw[start:end + 1]).get("unsupported", [])
        except (ValueError, json.JSONDecodeError):
            return []  # 판정자 출력이 깨지면 규칙 기반 결과만 사용
        return [f"LLM 판정 - 근거 부족: {u}" for u in unsupported]

    def run(self, summary: DischargeSummary, evidence: list[Document]) -> VerificationResult:
        issues = self.rule_checks(summary, evidence)
        if self.use_llm_judge:
            issues += self.llm_checks(summary, evidence)
        return VerificationResult(passed=not issues, issues=issues)
