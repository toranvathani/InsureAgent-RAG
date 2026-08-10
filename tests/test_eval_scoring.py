"""
Tests for the eval harness's field-scoring logic, isolated from any LLM calls.
"""
from app.eval.run_eval import _field_match


def test_critical_field_requires_exact_match():
    expected = {"claim_id": "CLM-1001"}
    actual = {"claim_id": "CLM-1002"}  # off by one digit
    correct, total, detail = _field_match(expected, actual)
    assert correct == 0
    assert total == 1
    assert detail[0]["method"] == "exact"


def test_critical_field_case_insensitive_exact_match():
    expected = {"policy_number": "PCY-88213"}
    actual = {"policy_number": "pcy-88213"}
    correct, total, detail = _field_match(expected, actual)
    assert correct == 1


def test_descriptive_field_accepts_elaboration_via_containment():
    expected = {"loss_type": "water damage"}
    actual = {"loss_type": "water damage from burst pipe"}
    correct, total, detail = _field_match(expected, actual)
    assert correct == 1
    assert detail[0]["method"].startswith("fuzzy")


def test_descriptive_field_rejects_unrelated_value():
    expected = {"loss_type": "water damage"}
    actual = {"loss_type": "theft"}
    correct, total, detail = _field_match(expected, actual)
    assert correct == 0


def test_mixed_critical_and_descriptive_fields():
    expected = {"claim_id": "CLM-1001", "loss_type": "fire"}
    actual = {"claim_id": "CLM-1001", "loss_type": "Fire damage to kitchen"}
    correct, total, detail = _field_match(expected, actual)
    assert correct == 2
    assert total == 2


def test_critical_numeric_field_int_vs_float_counts_as_match():
    # Regression test: JSON round-tripping often turns 8500 (int) into 8500.0
    # (float). These represent the same value and must not be scored as a
    # mismatch just because their string forms differ.
    expected = {"claim_amount": 8500}
    actual = {"claim_amount": 8500.0}
    correct, total, detail = _field_match(expected, actual)
    assert correct == 1


def test_critical_numeric_field_genuinely_different_values_fail():
    expected = {"claim_amount": 8500}
    actual = {"claim_amount": 8600}
    correct, total, detail = _field_match(expected, actual)
    assert correct == 0
