"""
Agent 2 — Policy Compliance / RAG Agent

Given an ExtractedClaim, retrieves relevant policy clauses from the vector
store and determines coverage alignment, deductible, and coverage limit,
always citing the retrieved source text (no unsupported claims).
"""
import json
import logging

from app.models.schemas import ComplianceResult, ExtractedClaim, PolicyCitation
from app.rag.retriever import retrieve_policy_context
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a P&C insurance policy compliance analyst.

You will be given: (1) structured claim details, and (2) relevant policy clauses
retrieved from the policy document store. Determine whether the claim appears to be
covered, and extract the applicable deductible and coverage limit IF stated in the
retrieved clauses.

Return ONLY valid JSON:
{
  "is_covered": true | false | null,
  "deductible": number or null,
  "coverage_limit": number or null,
  "citations": [{"source": string, "clause_text": string}],
  "notes": string
}

Rules:
- Base your answer ONLY on the retrieved clauses provided. Do not use outside knowledge.
- If the retrieved clauses don't clearly answer coverage, set is_covered to null and explain
  why in notes.
- Every citation must be a clause that actually appears in the retrieved context.
"""


def check_compliance(claim: ExtractedClaim, top_k: int = 5) -> ComplianceResult:
    query = f"{claim.loss_type or ''} {claim.loss_description}".strip()
    retrieved_chunks = retrieve_policy_context(query, top_k=top_k)

    context_block = "\n\n".join(
        f"[{c['source']}]: {c['text']}" for c in retrieved_chunks
    )

    user_content = (
        f"CLAIM DETAILS:\n{claim.model_dump_json(indent=2)}\n\n"
        f"RETRIEVED POLICY CLAUSES:\n{context_block or '(no relevant clauses found)'}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response_text = call_llm(messages)
    data = json.loads(response_text)
    citations = [PolicyCitation(**c) for c in data.get("citations", [])]

    return ComplianceResult(
        is_covered=data.get("is_covered"),
        deductible=data.get("deductible"),
        coverage_limit=data.get("coverage_limit"),
        citations=citations,
        notes=data.get("notes", ""),
    )
