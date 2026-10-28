"""HIS 연동을 가정한 REST API (FastAPI).

실행: uvicorn api.main:app --reload   (프로젝트 루트에서, PYTHONPATH=src)
문서: http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from medrag.config import get_settings
from medrag.graph import MedRAGPipeline
from medrag.qa import MedicalQA
from medrag.schema import DischargeSummary, to_fhir_composition

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    pipeline = MedRAGPipeline.from_settings(settings)
    state["pipeline"] = pipeline
    state["qa"] = MedicalQA(pipeline.retrieval.retriever, pipeline.llm, k=settings.top_k)
    yield
    state.clear()


app = FastAPI(title="MedRAG — 의료 문서 요약 API", version="0.1.0", lifespan=lifespan)


class SummarizeRequest(BaseModel):
    patient_id: str


class AskRequest(BaseModel):
    question: str
    patient_id: str | None = None


@app.get("/health")
def health():
    pipeline: MedRAGPipeline = state["pipeline"]
    return {"status": "ok", "llm": pipeline.llm.name}


@app.post("/summaries")
def create_summary(req: SummarizeRequest):
    try:
        result = state["pipeline"].summarize(req.patient_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "record_id": result.get("record_id"),
        "verified": result["passed"],
        "revisions": result["revisions"],
        "issues": result["issues"],
        "summary": result["summary"].model_dump(),
    }


@app.get("/summaries/{patient_id}")
def latest_summary(patient_id: str):
    record = state["pipeline"].store.latest(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail="저장된 요약이 없습니다.")
    return record


@app.get("/summaries/{patient_id}/fhir")
def latest_summary_fhir(patient_id: str):
    record = state["pipeline"].store.latest(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail="저장된 요약이 없습니다.")
    return to_fhir_composition(DischargeSummary.model_validate(record["summary"]))


@app.post("/ask")
def ask(req: AskRequest):
    answer = state["qa"].ask(req.question, patient_id=req.patient_id)
    return {"answer": answer.answer, "sources": answer.sources}
