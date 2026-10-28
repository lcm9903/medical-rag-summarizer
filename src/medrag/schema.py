"""구조화된 퇴원요약 스키마와 FHIR 변환.

LLM 출력을 자유 텍스트가 아닌 정해진 스키마(JSON)로 받으면
검증 에이전트가 항목별로 근거를 대조할 수 있고, HIS로 넘기기도 쉽습니다.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field, ValidationError


class Medication(BaseModel):
    name: str
    dose: str = ""
    frequency: str = ""


class DischargeSummary(BaseModel):
    patient_id: str
    chief_complaint: str = Field(description="주호소 1~2문장")
    diagnoses: list[str] = Field(default_factory=list)
    hospital_course: str = Field(description="입원 경과 요약 2~4문장")
    discharge_medications: list[Medication] = Field(default_factory=list)
    follow_up: str = Field(description="추적 관찰 계획")
    sources: list[str] = Field(default_factory=list, description="근거 청크 인용 (doc_id#section)")


def parse_summary_json(raw: str, patient_id: str) -> DischargeSummary:
    """LLM 응답에서 JSON 객체를 꺼내 스키마로 검증합니다.

    모델이 ```json 코드블록이나 앞뒤 설명을 붙여도 처리할 수 있도록
    첫 '{'부터 마지막 '}'까지를 잘라 파싱합니다.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("LLM 응답에서 JSON 객체를 찾지 못했습니다.")
    data = json.loads(cleaned[start:end + 1])
    data["patient_id"] = patient_id
    try:
        return DischargeSummary.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"요약 스키마 검증 실패: {exc}") from exc


# HL7 FHIR R4 Composition — 병원정보시스템(HIS) 연동용 표준 포맷 예시
_LOINC = {
    "document": ("18842-5", "Discharge summary"),
    "chief_complaint": ("10154-3", "Chief complaint"),
    "diagnoses": ("11535-2", "Hospital discharge Dx"),
    "hospital_course": ("8648-8", "Hospital course"),
    "medications": ("10183-2", "Hospital discharge medications"),
    "follow_up": ("18776-5", "Plan of care note"),
}


def _section(key: str, text: str) -> dict:
    code, display = _LOINC[key]
    return {
        "title": display,
        "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
        "text": {"status": "generated", "div": f"<div xmlns=\"http://www.w3.org/1999/xhtml\">{text}</div>"},
    }


def to_fhir_composition(summary: DischargeSummary, author: str = "MedRAG") -> dict:
    """요약을 FHIR Composition 리소스(dict)로 변환합니다. 실제 연동 시 기관 프로파일에 맞춰 확장합니다."""
    code, display = _LOINC["document"]
    meds = "; ".join(
        " ".join(p for p in (m.name, m.dose, m.frequency) if p) for m in summary.discharge_medications
    )
    return {
        "resourceType": "Composition",
        "status": "preliminary",  # 의료진 확인 전이므로 preliminary
        "type": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
        "subject": {"reference": f"Patient/{summary.patient_id}"},
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "author": [{"display": author}],
        "title": "AI 생성 퇴원요약 초안",
        "section": [
            _section("chief_complaint", summary.chief_complaint),
            _section("diagnoses", "; ".join(summary.diagnoses)),
            _section("hospital_course", summary.hospital_course),
            _section("medications", meds),
            _section("follow_up", summary.follow_up),
        ],
    }
