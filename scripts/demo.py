"""명령줄 데모: 환자 요약 생성 + 질의응답.

예) PYTHONPATH=src python scripts/demo.py --patient P001 --question "클로피도그렐은 언제 중단하나요?"
"""
from __future__ import annotations

import argparse
import json

from medrag.config import get_settings
from medrag.graph import MedRAGPipeline
from medrag.qa import MedicalQA
from medrag.schema import to_fhir_composition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="P001")
    parser.add_argument("--question", default=None)
    parser.add_argument("--fhir", action="store_true", help="FHIR Composition 형태로도 출력")
    parser.add_argument("--langgraph", action="store_true", help="LangGraph로 컴파일해 실행")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = MedRAGPipeline.from_settings(settings)
    print(f"LLM: {pipeline.llm.name} | 검색기: {settings.retriever}\n")

    if args.langgraph:
        result = pipeline.build_langgraph().invoke({"patient_id": args.patient})
    else:
        result = pipeline.summarize(args.patient)

    print("=== 퇴원요약 초안 ===")
    print(json.dumps(result["summary"].model_dump(), ensure_ascii=False, indent=2))
    print(f"\n검증 통과: {result['passed']} | 재작성 횟수: {result['revisions']}")
    for issue in result["issues"]:
        print(f"  - {issue}")

    if args.fhir:
        print("\n=== FHIR Composition ===")
        print(json.dumps(to_fhir_composition(result["summary"]), ensure_ascii=False, indent=2))

    if args.question:
        qa = MedicalQA(pipeline.retrieval.retriever, pipeline.llm, k=settings.top_k)
        answer = qa.ask(args.question, patient_id=args.patient)
        print(f"\n=== 질의응답 ===\nQ. {args.question}\nA. {answer.answer}")


if __name__ == "__main__":
    main()
