"""오프라인 단위 테스트 (외부 API·모델 없이 실행).

실행: PYTHONPATH=src python -m pytest tests   또는   PYTHONPATH=src python -m unittest discover tests
"""
from __future__ import annotations

import json
import unittest

from medrag.agents import VerifierAgent
from medrag.config import PROJECT_ROOT
from medrag.graph import MedRAGPipeline
from medrag.llm import DryRunLLM, parse_medication_line
from medrag.loader import load_notes, parse_note
from medrag.qa import MedicalQA
from medrag.retriever import TfidfRetriever
from medrag.schema import Medication, parse_summary_json, to_fhir_composition
from medrag.store import SummaryStore

NOTES = PROJECT_ROOT / "data" / "notes"


class FlakyLLM:
    """첫 요약에 가짜 약물을 넣고, 피드백을 받은 뒤에는 올바르게 다시 쓰는 테스트용 LLM."""

    name = "flaky"

    def __init__(self):
        self.inner = DryRunLLM()
        self.summary_calls = 0

    def invoke(self, prompt: str) -> str:
        out = self.inner.invoke(prompt)
        if "퇴원요약 작성을 돕는" in prompt:
            self.summary_calls += 1
            if "검증 에이전트가 다음 문제를 발견" not in prompt:
                data = json.loads(out)
                data["discharge_medications"].append({"name": "와파린(Warfarin)", "dose": "5 mg", "frequency": "1일 1회"})
                return json.dumps(data, ensure_ascii=False)
        return out


class LoaderTests(unittest.TestCase):
    def test_parse_front_matter_and_sections(self):
        meta, sections = parse_note((NOTES / "P001_discharge.md").read_text(encoding="utf-8"))
        self.assertEqual(meta["patient_id"], "P001")
        names = [s for s, _ in sections]
        self.assertIn("퇴원 약물", names)
        self.assertIn("주호소", names)

    def test_chunks_keep_metadata(self):
        docs = load_notes(NOTES)
        self.assertTrue(all("patient_id" in d.metadata and "section" in d.metadata for d in docs))
        self.assertEqual(len({d.metadata["doc_id"] for d in docs}), 6)

    def test_long_section_is_split_with_small_chunk_size(self):
        docs = load_notes(NOTES, chunk_size=60, chunk_overlap=10)
        course = [d for d in docs if d.metadata["doc_id"] == "P001-DS" and d.metadata["section"] == "입원 경과"]
        self.assertGreater(len(course), 1)


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.retriever = TfidfRetriever(load_notes(NOTES), k=4)

    def test_patient_filter_never_leaks_other_patients(self):
        for doc in self.retriever.invoke("퇴원 약물 아스피린", patient_id="P004", k=10):
            self.assertEqual(doc.metadata["patient_id"], "P004")

    def test_finds_relevant_section(self):
        docs = self.retriever.invoke("객담 배양 결과", patient_id="P002")
        self.assertEqual(docs[0].citation, "P002-DS#검사 소견")


class SchemaTests(unittest.TestCase):
    def test_parse_json_inside_code_block(self):
        raw = '설명입니다\n```json\n{"chief_complaint": "흉통", "diagnoses": ["NSTEMI"], "hospital_course": "스텐트", "follow_up": "2주 후"}\n```'
        summary = parse_summary_json(raw, "P004")
        self.assertEqual(summary.patient_id, "P004")
        self.assertEqual(summary.diagnoses, ["NSTEMI"])

    def test_medication_line_parsing(self):
        med = parse_medication_line("- 인슐린 글라진(Insulin glargine) 18 단위 | 1일 1회 취침 전 | 피하주사")
        self.assertEqual(med, {"name": "인슐린 글라진(Insulin glargine)", "dose": "18 단위", "frequency": "1일 1회 취침 전"})

    def test_fhir_composition(self):
        summary = parse_summary_json('{"chief_complaint": "a", "diagnoses": ["b"], "hospital_course": "c", "follow_up": "d"}', "P001")
        comp = to_fhir_composition(summary)
        self.assertEqual(comp["resourceType"], "Composition")
        self.assertEqual(comp["subject"]["reference"], "Patient/P001")
        self.assertEqual(len(comp["section"]), 5)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.retriever = TfidfRetriever(load_notes(NOTES), k=4)
        self.store = SummaryStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_dryrun_summary_is_verified_and_saved(self):
        pipeline = MedRAGPipeline(self.retriever, DryRunLLM(), self.store)
        result = pipeline.summarize("P004")
        self.assertTrue(result["passed"], result["issues"])
        names = [m.name for m in result["summary"].discharge_medications]
        self.assertIn("티카그렐러(Ticagrelor)", names)
        self.assertEqual(self.store.latest("P004")["id"], result["record_id"])

    def test_verifier_triggers_revision_and_fixes_hallucination(self):
        llm = FlakyLLM()
        pipeline = MedRAGPipeline(self.retriever, llm, self.store, max_revisions=1)
        result = pipeline.summarize("P001")
        self.assertEqual(llm.summary_calls, 2)
        self.assertEqual(result["revisions"], 1)
        self.assertTrue(result["passed"])
        self.assertNotIn("와파린(Warfarin)", [m.name for m in result["summary"].discharge_medications])

    def test_unfixed_hallucination_is_saved_as_unverified(self):
        llm = FlakyLLM()
        pipeline = MedRAGPipeline(self.retriever, llm, self.store, max_revisions=0)
        result = pipeline.summarize("P001")
        self.assertFalse(result["passed"])
        self.assertFalse(self.store.latest("P001")["verified"])

    def test_verifier_catches_wrong_dose_and_number(self):
        pipeline = MedRAGPipeline(self.retriever, DryRunLLM(), None)
        result = pipeline.summarize("P002")
        bad = result["summary"].model_copy(deep=True)
        bad.discharge_medications[0] = Medication(name=bad.discharge_medications[0].name, dose="1000 mg")
        bad.hospital_course += " 퇴원 시 CRP 0.3 mg/dL."
        issues = VerifierAgent().run(bad, result["evidence"]).issues
        self.assertTrue(any("용량" in i for i in issues))
        self.assertTrue(any("수치" in i for i in issues))

    def test_unknown_patient_raises(self):
        pipeline = MedRAGPipeline(self.retriever, DryRunLLM(), None)
        with self.assertRaises(LookupError):
            pipeline.summarize("P999")

    def test_qa_returns_citation(self):
        answer = MedicalQA(self.retriever, DryRunLLM()).ask("클로피도그렐은 언제 중단하나요?", patient_id="P001")
        self.assertIn("[P001-OP#입원 경과]", answer.answer)

    def test_langgraph_matches_plain_loop(self):
        try:
            import langgraph  # noqa: F401
        except ImportError:
            self.skipTest("langgraph 미설치")
        pipeline = MedRAGPipeline(self.retriever, DryRunLLM(), None)
        result = pipeline.build_langgraph().invoke({"patient_id": "P003"})
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
