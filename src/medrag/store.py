"""요약 결과 저장소 (SQLite).

AI가 만든 요약은 의료진 검토 전까지 '초안'이므로,
검증 통과 여부와 발견된 문제, 사용한 모델을 함께 기록해 추적 가능하게 둡니다.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .schema import DischargeSummary

SCHEMA = """
CREATE TABLE IF NOT EXISTS summaries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    model        TEXT    NOT NULL,
    verified     INTEGER NOT NULL,
    revisions    INTEGER NOT NULL DEFAULT 0,
    issues_json  TEXT    NOT NULL,
    summary_json TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_patient ON summaries (patient_id, created_at);
"""


class SummaryStore:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def save(self, summary: DischargeSummary, model: str, verified: bool, issues: list[str], revisions: int = 0) -> int:
        cur = self.conn.execute(
            "INSERT INTO summaries (patient_id, created_at, model, verified, revisions, issues_json, summary_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                summary.patient_id,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                model,
                int(verified),
                revisions,
                json.dumps(issues, ensure_ascii=False),
                summary.model_dump_json(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def latest(self, patient_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM summaries WHERE patient_id = ? ORDER BY id DESC LIMIT 1", (patient_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def history(self, patient_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM summaries WHERE patient_id = ? ORDER BY id DESC", (patient_id,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(verified), 0) AS verified, "
            "COALESCE(AVG(revisions), 0) AS avg_revisions FROM summaries"
        ).fetchone()
        return dict(row)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "patient_id": row["patient_id"],
            "created_at": row["created_at"],
            "model": row["model"],
            "verified": bool(row["verified"]),
            "revisions": row["revisions"],
            "issues": json.loads(row["issues_json"]),
            "summary": json.loads(row["summary_json"]),
        }
