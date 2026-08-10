"""
Unit tests for Pydantic schema validation — no LLM calls, fast and deterministic.
Run: pytest tests/ -v
"""
import pytest
from pydantic import ValidationError

from app.models.schemas import (
    ClaimDecision,
    ClaimStatus,
    ComplianceResult,
    ExtractedClaim,
)


def test_extracted_claim_valid():
    claim = ExtractedClaim(
        claim_id="CLM-1001",
        policyholder_name="Rajesh Kumar",
        policy_number="PCY-88213",
        loss_date="2026-05-12",
        claim_amount=8500,
        loss_description="Kitchen fire caused smoke damage.",
        loss_type="fire",
    )
    assert claim.claim_amount == 8500
    assert claim.missing_fields == []


def test_extracted_claim_rejects_negative_amount():
    with pytest.raises(ValidationError):
        ExtractedClaim(
            claim_id="CLM-1002",
            policyholder_name="Test",
            policy_number="PCY-1",
            loss_description="test",
            claim_amount=-500,
        )


def test_compliance_result_defaults():
    result = ComplianceResult()
    assert result.is_covered is None
    assert result.citations == []


def test_claim_decision_confidence_bounds():
    with pytest.raises(ValidationError):
        ClaimDecision(
            claim_id="CLM-1001",
            status=ClaimStatus.APPROVED,
            confidence=1.5,  # invalid, must be <= 1
            reasoning="test",
        )


def test_claim_decision_valid_status_enum():
    decision = ClaimDecision(
        claim_id="CLM-1001",
        status="FLAGGED_FOR_FRAUD",
        confidence=0.82,
        reasoning="High value theft claim filed shortly after policy inception.",
    )
    assert decision.status == ClaimStatus.FLAGGED_FOR_FRAUD
