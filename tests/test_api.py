"""API 테스트 (fastapi, httpx 설치 시 실행)."""
from __future__ import annotations

import os
import tempfile
import unittest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None


@unittest.skipIf(TestClient is None, "fastapi/httpx 미설치")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ.update(LLM_PROVIDER="dryrun", RETRIEVER="tfidf", DB_PATH=f"{self.tmp.name}/test.db")
        from api.main import app

        self.client_cm = TestClient(app)
        self.client = self.client_cm.__enter__()

    def tearDown(self):
        self.client_cm.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_summary_roundtrip_and_fhir(self):
        created = self.client.post("/summaries", json={"patient_id": "P002"}).json()
        self.assertTrue(created["verified"])
        latest = self.client.get("/summaries/P002").json()
        self.assertEqual(latest["id"], created["record_id"])
        fhir = self.client.get("/summaries/P002/fhir").json()
        self.assertEqual(fhir["resourceType"], "Composition")

    def test_unknown_patient_404(self):
        self.assertEqual(self.client.post("/summaries", json={"patient_id": "P999"}).status_code, 404)

    def test_ask(self):
        body = self.client.post("/ask", json={"question": "객담 배양 결과는?", "patient_id": "P002"}).json()
        self.assertIn("P002-DS#검사 소견", body["sources"])


if __name__ == "__main__":
    unittest.main()
