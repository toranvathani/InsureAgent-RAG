"""
Agent 3 — Decision Agent

Synthesizes the Extraction Agent and Compliance Agent outputs into a final
structured recommendation for a human adjuster.
"""
import json
import logging

from app.models.schemas import (
    ClaimDecision,
    ComplianceResult,
    ExtractedClaim,
)
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior insurance claims decision-support assistant.

You will receive the extracted claim data and the policy compliance analysis. Produce a
final recommendation for a human adjuster. Return ONLY valid JSON:

{
  "status": "APPROVED" | "FLAGGED_FOR_FRAUD" | "MORE_INFO_NEEDED" | "DENIED",
  "confidence": number between 0 and 1,
  "reasoning": string (2-4 sentences, plain language, for a human adjuster),
  "missing_fields": array of strings
}

Decision rules (apply in this order):
1. If required fields are missing -> MORE_INFO_NEEDED.
2. Check for fraud signals FIRST, before considering coverage: claim amount that
   dramatically exceeds a stated coverage cap, a claim filed within days of policy
   inception, missing documentation the policy requires (e.g. no police report for
   theft), or inconsistent/vague details. If two or more such signals are present
   -> FLAGGED_FOR_FRAUD, even if the compliance analysis also says "not covered" —
   a claim that is both suspicious AND technically uncovered should be flagged for
   human review, not silently denied.
3. If no fraud signals, and compliance analysis says not covered -> DENIED.
4. Otherwise, if covered and complete -> APPROVED.
- Never state a fact that isn't supported by the provided extraction or compliance data.
"""


def make_decision(
    claim: ExtractedClaim, compliance: ComplianceResult
) -> ClaimDecision:
    user_content = (
        f"EXTRACTED CLAIM:\n{claim.model_dump_json(indent=2)}\n\n"
        f"COMPLIANCE ANALYSIS:\n{compliance.model_dump_json(indent=2)}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response_text = call_llm(messages)
    data = json.loads(response_text)

    return ClaimDecision(
        claim_id=claim.claim_id,
        status=data["status"],
        confidence=data["confidence"],
        reasoning=data["reasoning"],
        missing_fields=data.get("missing_fields", claim.missing_fields),
        citations=compliance.citations,
    )
