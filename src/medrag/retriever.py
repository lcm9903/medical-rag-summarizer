"""검색기(Retriever).

- TfidfRetriever: 문자 n-gram TF-IDF. 외부 모델 없이 동작하는 오프라인 기준선이며,
  한국어처럼 띄어쓰기·조사 변형이 많은 텍스트에서 단어 단위보다 견고합니다.
- VectorRetriever: LangChain FAISS + 한국어 문장 임베딩(HuggingFace).

두 검색기 모두 `invoke(query, patient_id=None)` 인터페이스를 공유하므로
파이프라인 코드는 어떤 검색기를 쓰는지 몰라도 됩니다.
환자 ID 필터는 다른 환자의 기록이 요약에 섞이는 것을 막는 안전장치입니다.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .loader import Document


class Retriever(Protocol):
    def invoke(self, query: str, patient_id: str | None = None, k: int | None = None) -> list[Document]: ...


class TfidfRetriever:
    def __init__(self, docs: list[Document], k: int = 4):
        if not docs:
            raise ValueError("검색할 문서가 없습니다.")
        self.docs = docs
        self.k = k
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        # 섹션명을 본문 앞에 붙여 "퇴원 약물" 같은 질의가 해당 섹션과 잘 맞도록 함
        corpus = [f"{d.metadata.get('section', '')} {d.page_content}" for d in docs]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def invoke(self, query: str, patient_id: str | None = None, k: int | None = None) -> list[Document]:
        k = k or self.k
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        if patient_id is not None:
            mask = np.array([d.metadata.get("patient_id") == patient_id for d in self.docs])
            scores = np.where(mask, scores, -1.0)
        order = np.argsort(-scores)
        return [self.docs[i] for i in order[:k] if scores[i] > 0]


class VectorRetriever:
    """LangChain FAISS 벡터스토어 기반 검색기 (langchain-community, langchain-huggingface 필요)."""

    def __init__(self, docs: list[Document], embedding_model: str, k: int = 4):
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model, encode_kwargs={"normalize_embeddings": True}
        )
        self.k = k
        self.store = FAISS.from_documents([d.to_langchain() for d in docs], embeddings)

    def invoke(self, query: str, patient_id: str | None = None, k: int | None = None) -> list[Document]:
        k = k or self.k
        kwargs = {"k": k}
        if patient_id is not None:
            # 필터는 검색 후 적용되므로 후보를 넉넉히 가져온 뒤 거릅니다.
            kwargs.update(filter={"patient_id": patient_id}, fetch_k=max(50, k * 10))
        results = self.store.similarity_search(query, **kwargs)
        return [Document(page_content=r.page_content, metadata=dict(r.metadata)) for r in results]


def build_retriever(docs: list[Document], kind: str = "tfidf", embedding_model: str = "", k: int = 4) -> Retriever:
    if kind == "tfidf":
        return TfidfRetriever(docs, k=k)
    if kind == "faiss":
        return VectorRetriever(docs, embedding_model=embedding_model, k=k)
    raise ValueError(f"알 수 없는 검색기: {kind}")
