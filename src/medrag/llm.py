"""LLM 백엔드.

파이프라인은 `invoke(prompt) -> str` 하나만 기대합니다.
- openai / ollama: LangChain 채팅 모델(ChatOpenAI, ChatOllama)을 감싸서 사용
- dryrun: API 키·GPU 없이 파이프라인 전체를 실행·테스트하기 위한 규칙 기반 추출기.
  LLM 성능을 대신하는 것이 아니라, 검색·검증·저장·API 흐름을 검증하기 위한 용도입니다.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from .prompts import CONTEXT_END, CONTEXT_START


class LLM(Protocol):
    name: str

    def invoke(self, prompt: str) -> str: ...


class LangChainChat:
    """LangChain 채팅 모델 래퍼."""

    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def invoke(self, prompt: str) -> str:
        return self.model.invoke(prompt).content


_CITE = re.compile(r"^\[([^\[\]#]+#[^\[\]]+)\]\s?(.*)$")
_DOSE = re.compile(r"^(.*?)\s+(\d[\d.,/]*\s*(?:mg|g|mcg|μg|IU|단위|mL|L))\s*$")


def parse_context(prompt: str) -> list[tuple[str, str]]:
    """프롬프트의 CONTEXT 블록을 [(인용ID, 본문), ...]으로 복원."""
    if CONTEXT_START not in prompt:
        return []
    block = prompt.split(CONTEXT_START, 1)[1].split(CONTEXT_END, 1)[0]
    entries: list[list[str]] = []
    for line in block.strip().splitlines():
        m = _CITE.match(line)
        if m:
            entries.append([m.group(1), m.group(2)])
        elif entries:
            entries[-1][1] += "\n" + line
    return [(c, t.strip()) for c, t in entries]


def parse_medication_line(line: str) -> dict | None:
    """'- 아스피린(Aspirin) 100 mg | 1일 1회 | 경구' 형태의 줄을 약물 dict로 변환."""
    line = line.strip().lstrip("-").strip()
    if not line:
        return None
    parts = [p.strip() for p in line.split("|")]
    m = _DOSE.match(parts[0])
    name, dose = (m.group(1).strip(), m.group(2).strip()) if m else (parts[0], "")
    frequency = parts[1] if len(parts) > 1 else ""
    return {"name": name, "dose": dose, "frequency": frequency}


def _first_sentences(text: str, n: int) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
    return " ".join(sentences[:n]).strip()


class DryRunLLM:
    """CONTEXT에서 섹션을 그대로 추출해 JSON을 만드는 결정론적 백엔드 (오프라인 테스트용)."""

    name = "dryrun"

    def invoke(self, prompt: str) -> str:
        context = parse_context(prompt)
        if "사실성을 검토" in prompt:
            return json.dumps({"unsupported": []}, ensure_ascii=False)
        if "질의응답" in prompt:
            return self._answer(context)
        return self._summarize(context)

    @staticmethod
    def _pick(context: list[tuple[str, str]], section: str) -> tuple[str, str] | None:
        hits = [(c, t) for c, t in context if c.split("#", 1)[1] == section]
        # 같은 섹션이 여러 문서에 있으면 퇴원요약(-DS)을 우선
        hits.sort(key=lambda ct: 0 if "-DS#" in ct[0] else 1)
        return hits[0] if hits else None

    def _summarize(self, context: list[tuple[str, str]]) -> str:
        used: list[str] = []

        def take(section: str) -> str:
            hit = self._pick(context, section)
            if not hit:
                return ""
            used.append(hit[0])
            return hit[1]

        diagnoses_raw = take("진단명")
        meds_raw = take("퇴원 약물")
        result = {
            "chief_complaint": _first_sentences(take("주호소"), 1),
            "diagnoses": [l.strip().lstrip("-").strip() for l in diagnoses_raw.splitlines() if l.strip()],
            "hospital_course": _first_sentences(take("입원 경과"), 3),
            "discharge_medications": [
                med for l in meds_raw.splitlines() if (med := parse_medication_line(l))
            ],
            "follow_up": _first_sentences(take("추적 관찰 계획"), 2),
            "sources": used,
        }
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _answer(context: list[tuple[str, str]]) -> str:
        if not context:
            return "기록에서 확인할 수 없습니다."
        cite, text = context[0]
        return f"{_first_sentences(text, 2)} [{cite}]"


def get_llm(provider: str, openai_model: str = "gpt-4o-mini", ollama_model: str = "qwen2.5:7b") -> LLM:
    if provider == "dryrun":
        return DryRunLLM()
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # OPENAI_API_KEY 환경 변수 필요

        return LangChainChat(ChatOpenAI(model=openai_model, temperature=0), name=f"openai:{openai_model}")
    if provider == "ollama":
        from langchain_ollama import ChatOllama  # 로컬 Ollama 서버 필요

        return LangChainChat(ChatOllama(model=ollama_model, temperature=0), name=f"ollama:{ollama_model}")
    raise ValueError(f"알 수 없는 LLM 제공자: {provider}")
