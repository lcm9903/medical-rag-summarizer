"""평가 스크립트.

1. 검색 성능   : 질문별 정답 문서·섹션이 상위 k개 안에 있는지 (Hit@1, Hit@k, MRR)
2. 요약 검증률 : 전체 환자 요약 중 검증 에이전트를 통과한 비율
3. 환각 탐지율 : 정상 요약에 가짜 약물·틀린 용량·없는 수치를 주입했을 때 검증 에이전트가 잡아내는 비율

예) PYTHONPATH=src python scripts/evaluate.py
"""
from __future__ import annotations

import json
from pathlib import Path

from medrag.agents import VerifierAgent
from medrag.config import PROJECT_ROOT, get_settings
from medrag.graph import MedRAGPipeline
from medrag.schema import Medication

EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "questions.json"
OUT_DIR = PROJECT_ROOT / "outputs"


def evaluate_retrieval(retriever, questions: list[dict], k: int) -> dict:
    hit1 = hitk = rr = 0.0
    rows = []
    for q in questions:
        docs = retriever.invoke(q["question"], patient_id=q["patient_id"], k=k)
        target = f"{q['expected_doc']}#{q['expected_section']}"
        cites = [d.citation for d in docs]
        rank = cites.index(target) + 1 if target in cites else None
        hit1 += rank == 1
        hitk += rank is not None
        rr += 1 / rank if rank else 0
        rows.append({"question": q["question"], "target": target, "rank": rank, "retrieved": cites})
    n = len(questions)
    return {"n": n, f"hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": rr / n, "details": rows}


def inject_errors(summary):
    """정상 요약에서 세 종류의 오류 요약을 만듭니다."""
    fake_med = summary.model_copy(deep=True)
    fake_med.discharge_medications.append(Medication(name="와파린(Warfarin)", dose="5 mg", frequency="1일 1회"))

    wrong_dose = summary.model_copy(deep=True)
    if wrong_dose.discharge_medications:
        wrong_dose.discharge_medications[0].dose = "999 mg"

    fake_number = summary.model_copy(deep=True)
    fake_number.hospital_course += " 퇴원 시 좌심실 구혈률 73%로 회복됨."

    return {"가짜 약물": fake_med, "틀린 용량": wrong_dose, "없는 수치": fake_number}


def main() -> None:
    settings = get_settings()
    settings.db_path = Path(":memory:")
    pipeline = MedRAGPipeline.from_settings(settings)
    retriever = pipeline.retrieval.retriever
    questions = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    retrieval = evaluate_retrieval(retriever, questions, settings.top_k)

    patients = sorted({d.metadata["patient_id"] for d in retriever.docs}) if hasattr(retriever, "docs") \
        else sorted({q["patient_id"] for q in questions if q["patient_id"]})
    verifier = VerifierAgent(llm=None, use_llm_judge=False)  # 규칙 기반 검증만으로 탐지율 측정

    passed = 0
    detection: dict[str, list[bool]] = {}
    for pid in patients:
        result = pipeline.summarize(pid)
        passed += result["passed"]
        for kind, bad in inject_errors(result["summary"]).items():
            detected = not verifier.run(bad, result["evidence"]).passed
            detection.setdefault(kind, []).append(detected)

    report = {
        "llm": pipeline.llm.name,
        "retriever": settings.retriever,
        "retrieval": retrieval,
        "summary_verification_rate": passed / len(patients),
        "hallucination_detection": {k: sum(v) / len(v) for k, v in detection.items()},
        "n_patients": len(patients),
    }

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    k = settings.top_k
    print(f"LLM: {report['llm']} | 검색기: {report['retriever']}")
    print(f"[검색] 질문 {retrieval['n']}개 | Hit@1 {retrieval['hit@1']:.2f} | Hit@{k} {retrieval[f'hit@{k}']:.2f} | MRR {retrieval['mrr']:.2f}")
    print(f"[요약] 검증 통과율 {report['summary_verification_rate']:.2f} ({len(patients)}명)")
    for kind, rate in report["hallucination_detection"].items():
        print(f"[환각 탐지] {kind}: {rate:.2f}")
    misses = [r for r in retrieval["details"] if r["rank"] != 1]
    if misses:
        print("\n1위로 찾지 못한 질문:")
        for r in misses:
            print(f"  - {r['question']} → 정답 {r['target']}, 순위 {r['rank']}")


if __name__ == "__main__":
    main()
