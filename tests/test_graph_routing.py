"""
Tests for the graph's conditional routing logic. These don't call any LLM —
they just verify that route_after_extract sends incomplete claims down the
short-circuit path and complete claims to the compliance agent.
"""
from app.graph import route_after_extract
from app.models.schemas import ExtractedClaim


def _make_claim(**overrides) -> ExtractedClaim:
    defaults = dict(
        claim_id="CLM-1001",
        policyholder_name="Rajesh Kumar",
        policy_number="PCY-88213",
        loss_description="Kitchen fire caused smoke damage.",
    )
    defaults.update(overrides)
    return ExtractedClaim(**defaults)


def test_routes_to_compliance_when_all_required_fields_present():
    claim = _make_claim()
    state = {"extraction": claim}
    assert route_after_extract(state) == "compliance"


def test_routes_to_needs_more_info_when_claim_id_missing():
    claim = _make_claim(claim_id="", missing_fields=["claim_id"])
    state = {"extraction": claim}
    assert route_after_extract(state) == "needs_more_info"


def test_routes_to_needs_more_info_when_policy_number_flagged_missing():
    claim = _make_claim(missing_fields=["policy_number"])
    state = {"extraction": claim}
    assert route_after_extract(state) == "needs_more_info"
