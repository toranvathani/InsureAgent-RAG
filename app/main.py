"""
InsureAgent-RAG API — orchestrates the 3-agent claims pipeline.
"""
import logging
import time

from fastapi import FastAPI, HTTPException

from app.graph import run_claim_pipeline
from app.models.schemas import ProcessClaimRequest, ProcessClaimResponse
from app.services import db, monitoring

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="InsureAgent-RAG",
    description="Autonomous P&C Insurance Claims Copilot — Multi-Agent RAG System",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    import time as _time

    try:
        db.init_db()
    except Exception as e:
        logger.warning("DB init skipped/failed (ok for local dev without Postgres): %s", e)

    # Warm the embedding model now instead of on the first request — sentence-transformers
    # loads its weights from disk on first use, which otherwise adds a few seconds of
    # latency to whichever request happens to be first.
    try:
        from app.rag.retriever import _get_embedder

        t0 = _time.time()
        _get_embedder()
        logger.info("Embedding model warmed in %.1fs", _time.time() - t0)
    except Exception as e:
        logger.warning("Embedding model warm-up skipped/failed: %s", e)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-claim", response_model=ProcessClaimResponse)
def process_claim(request: ProcessClaimRequest):
    start = time.time()
    try:
        final_state = run_claim_pipeline(request.raw_document_text)
        extraction = final_state["extraction"]
        compliance = final_state["compliance"]
        decision = final_state["decision"]

        try:
            db.save_claim_result(
                extraction.claim_id,
                extraction.model_dump(mode="json"),
                compliance.model_dump(mode="json"),
                decision.model_dump(mode="json"),
            )
        except Exception as e:
            logger.warning("Could not persist claim result: %s", e)

        monitoring.record_request("/process-claim", (time.time() - start) * 1000, success=True)
        return ProcessClaimResponse(extraction=extraction, compliance=compliance, decision=decision)

    except Exception as e:
        monitoring.record_request("/process-claim", (time.time() - start) * 1000, success=False)
        logger.exception("Claim processing failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/claims/{claim_id}")
def get_claim(claim_id: str):
    result = db.get_claim_result(claim_id)
    if not result:
        raise HTTPException(status_code=404, detail="Claim not found")
    return result


@app.get("/metrics")
def metrics():
    return monitoring.get_metrics()


@app.get("/eval-report")
def eval_report():
    from app.eval.run_eval import run_eval

    return run_eval()
