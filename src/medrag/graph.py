"""에이전트 오케스트레이션.

    retrieve → summarize → verify ─(통과 또는 재시도 소진)→ save
                   ↑            │
                   └──(실패: 문제 목록 피드백)──┘

같은 노드 함수를 두 방식으로 실행할 수 있습니다.
- `MedRAGPipeline.summarize()` : 순수 Python 루프 (의존성 없음, 테스트용)
- `MedRAGPipeline.build_langgraph()` : LangGraph StateGraph로 컴파일 (langgraph 설치 시)
"""
from __future__ import annotations

from typing import Any, TypedDict

from .agents import RetrievalAgent, SummarizerAgent, VerifierAgent
from .config import Settings
from .llm import LLM, get_llm
from .loader import Document, load_notes
from .retriever import Retriever, build_retriever
from .schema import DischargeSummary
from .store import SummaryStore


class PipelineState(TypedDict, total=False):
    patient_id: str
    evidence: list[Document]
    summary: DischargeSummary
    issues: list[str]
    passed: bool
    revisions: int
    record_id: int


class MedRAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLM, store: SummaryStore | None = None,
                 max_revisions: int = 1, use_llm_judge: bool = True):
        self.llm = llm
        self.store = store
        self.max_revisions = max_revisions
        self.retrieval = RetrievalAgent(retriever)
        self.summarizer = SummarizerAgent(llm)
        self.verifier = VerifierAgent(llm, use_llm_judge=use_llm_judge)

    @classmethod
    def from_settings(cls, settings: Settings) -> "MedRAGPipeline":
        docs = load_notes(settings.notes_dir, settings.chunk_size, settings.chunk_overlap)
        retriever = build_retriever(docs, settings.retriever, settings.embedding_model, settings.top_k)
        llm = get_llm(settings.llm_provider, settings.openai_model, settings.ollama_model)
        return cls(retriever, llm, SummaryStore(settings.db_path), settings.max_revisions)

    # ---- 노드 -------------------------------------------------------------
    def node_retrieve(self, state: PipelineState) -> dict[str, Any]:
        evidence = self.retrieval.run(state["patient_id"])
        if not evidence:
            raise LookupError(f"환자 {state['patient_id']}의 기록을 찾지 못했습니다.")
        return {"evidence": evidence, "revisions": 0, "issues": []}

    def node_summarize(self, state: PipelineState) -> dict[str, Any]:
        revising = bool(state.get("issues"))
        summary = self.summarizer.run(state["patient_id"], state["evidence"], state.get("issues"))
        return {"summary": summary, "revisions": state.get("revisions", 0) + int(revising)}

    def node_verify(self, state: PipelineState) -> dict[str, Any]:
        result = self.verifier.run(state["summary"], state["evidence"])
        return {"passed": result.passed, "issues": result.issues}

    def route_after_verify(self, state: PipelineState) -> str:
        if state["passed"] or state.get("revisions", 0) >= self.max_revisions:
            return "save"
        return "revise"

    def node_save(self, state: PipelineState) -> dict[str, Any]:
        if self.store is None:
            return {}
        record_id = self.store.save(
            state["summary"], model=self.llm.name, verified=state["passed"],
            issues=state.get("issues", []), revisions=state.get("revisions", 0),
        )
        return {"record_id": record_id}

    # ---- 실행 -------------------------------------------------------------
    def summarize(self, patient_id: str) -> PipelineState:
        state: PipelineState = {"patient_id": patient_id}
        state.update(self.node_retrieve(state))
        while True:
            state.update(self.node_summarize(state))
            state.update(self.node_verify(state))
            if self.route_after_verify(state) == "save":
                break
        state.update(self.node_save(state))
        return state

    def build_langgraph(self):
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(PipelineState)
        graph.add_node("retrieve", self.node_retrieve)
        graph.add_node("summarize", self.node_summarize)
        graph.add_node("verify", self.node_verify)
        graph.add_node("save", self.node_save)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "summarize")
        graph.add_edge("summarize", "verify")
        graph.add_conditional_edges("verify", self.route_after_verify, {"revise": "summarize", "save": "save"})
        graph.add_edge("save", END)
        return graph.compile()


def summary_to_dict(summary: DischargeSummary) -> dict:
    return summary.model_dump()
