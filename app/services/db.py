"""
PostgreSQL persistence for claim processing audit logs / metadata.

Uses the shared connection pool (app/services/db_pool.py) instead of opening
a fresh connection per call, which matters a lot against Neon's serverless
compute — a pooled connection avoids repeated SSL handshake + potential
cold-start overhead on every single request.
"""
import json

from app.services.db_pool import pooled_conn


def init_db():
    with pooled_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS claim_decisions (
                id SERIAL PRIMARY KEY,
                claim_id TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence FLOAT,
                extraction JSONB,
                compliance JSONB,
                decision JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
        )
        conn.commit()
        cur.close()


def save_claim_result(claim_id: str, extraction: dict, compliance: dict, decision: dict):
    with pooled_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO claim_decisions (claim_id, status, confidence, extraction, compliance, decision)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                claim_id,
                decision["status"],
                decision["confidence"],
                json.dumps(extraction),
                json.dumps(compliance),
                json.dumps(decision),
            ),
        )
        conn.commit()
        cur.close()


def get_claim_result(claim_id: str) -> dict | None:
    with pooled_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT claim_id, status, confidence, extraction, compliance, decision, created_at
            FROM claim_decisions WHERE claim_id = %s
            ORDER BY created_at DESC LIMIT 1
            """,
            (claim_id,),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        return {
            "claim_id": row[0],
            "status": row[1],
            "confidence": row[2],
            "extraction": row[3],
            "compliance": row[4],
            "decision": row[5],
            "created_at": row[6].isoformat(),
        }
