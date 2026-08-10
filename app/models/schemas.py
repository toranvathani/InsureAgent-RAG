"""
Pydantic schemas enforcing structured LLM outputs across all three agents.
"""
from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------- Agent 1: Extraction ----------

class ExtractedClaim(BaseModel):
    claim_id: str = Field(..., description="Unique claim identifier")
    policyholder_name: str
    policy_number: str
    loss_date: Optional[date] = Field(None, description="Date the loss occurred")
    claim_amount: Optional[float] = Field(None, ge=0)
    loss_description: str
    loss_type: Optional[str] = Field(
        None, description="e.g. fire, theft, water damage, collision"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Fields the extraction agent could not confidently populate",
    )


# ---------- Agent 2: Policy Compliance / RAG ----------

class PolicyCitation(BaseModel):
    source: str = Field(..., description="Document/section the clause was retrieved from")
    clause_text: str


class ComplianceResult(BaseModel):
    is_covered: Optional[bool] = Field(
        None, description="None if inconclusive from retrieved context"
    )
    deductible: Optional[float] = None
    coverage_limit: Optional[float] = None
    citations: List[PolicyCitation] = Field(default_factory=list)
    notes: str = ""


# ---------- Agent 3: Decision ----------

class ClaimStatus(str, Enum):
    APPROVED = "APPROVED"
    FLAGGED_FOR_FRAUD = "FLAGGED_FOR_FRAUD"
    MORE_INFO_NEEDED = "MORE_INFO_NEEDED"
    DENIED = "DENIED"


class ClaimDecision(BaseModel):
    claim_id: str
    status: ClaimStatus
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str
    missing_fields: List[str] = Field(default_factory=list)
    citations: List[PolicyCitation] = Field(default_factory=list)


# ---------- API request/response wrappers ----------

class ProcessClaimRequest(BaseModel):
    raw_document_text: str = Field(..., description="Raw FNOL/claim note/policy text")


class ProcessClaimResponse(BaseModel):
    extraction: ExtractedClaim
    compliance: ComplianceResult
    decision: ClaimDecision
