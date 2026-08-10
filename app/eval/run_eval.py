"""
Evaluation harness: runs the pipeline against a labeled eval set and scores
field-level extraction accuracy + decision accuracy. Wired into CI so prompt
changes can't silently regress quality.

Scoring strategy:
- CRITICAL fields (claim_id, policy_number, claim_amount, loss_date, policyholder_name)
  are scored with exact match — precision matters here, a wrong ID or amount is a real bug.
- DESCRIPTIVE fields (loss_type, loss_description) are scored with fuzzy matching, since
  "fire" vs "Fire damage" or a differently-phrased-but-correct summary shouldn't count as
  a failure. Cosmetic mismatches on these fields were previously inflating the "accuracy
  gap" without reflecting an actual extraction problem.

Usage:
    python -m app.eval.run_eval
"""
import json
from difflib import SequenceMatcher
from pathlib import Path

from app.graph import run_claim_pipeline

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"

CRITICAL_FIELDS = {"claim_id", "policy_number", "claim_amount", "loss_date", "policyholder_name"}
FUZZY_MATCH_THRESHOLD = 0.6  # similarity ratio above which a descriptive field counts as correct


def _normalize(value) -> str:
    if isinstance(value, (int, float)):
        # Normalize numeric types so 8500 and 8500.0 compare equal — they're the
        # same value, just different Python/JSON numeric representations, and
        # treating them as a mismatch was a scorer bug, not an extraction bug.
        return str(float(value))
    return str(value).strip().lower()


def _fuzzy_match(expected_norm: str, actual_norm: str) -> tuple[bool, float]:
    """
    Returns (is_match, similarity_score). Combines two signals:
    1. Containment — catches cases like expected="fire" vs actual="fire damage to
       kitchen", where the model gave a correct but more elaborate answer. Pure
       character-ratio scoring penalizes this unfairly since the length delta
       dominates the ratio even though the content is correct.
    2. SequenceMatcher ratio — catches close-but-not-identical phrasing where
       neither string contains the other (e.g. "water damage" vs "water leak").
    """
    if not expected_norm or not actual_norm:
        return False, 0.0
    if expected_norm in actual_norm or actual_norm in expected_norm:
        return True, 1.0
    ratio = SequenceMatcher(None, expected_norm, actual_norm).ratio()
    return ratio >= FUZZY_MATCH_THRESHOLD, ratio


def _field_match(expected: dict, actual: dict) -> tuple[int, int, list[dict]]:
    """Returns (correct_count, total_count, per-field detail for debugging)."""
    correct, total = 0, 0
    detail = []

    for key, expected_val in expected.items():
        total += 1
        actual_val = actual.get(key)
        expected_norm = _normalize(expected_val)
        actual_norm = _normalize(actual_val)

        if key in CRITICAL_FIELDS:
            is_match = expected_norm == actual_norm
            method = "exact"
        else:
            is_match, ratio = _fuzzy_match(expected_norm, actual_norm)
            method = f"fuzzy({ratio:.2f})"

        if is_match:
            correct += 1

        detail.append(
            {
                "field": key,
                "expected": expected_val,
                "actual": actual_val,
                "match": is_match,
                "method": method,
            }
        )

    return correct, total, detail


def run_eval() -> dict:
    cases = json.loads(EVAL_SET_PATH.read_text())

    total_field_correct = 0
    total_fields = 0
    decision_correct = 0
    results = []

    for case in cases:
        final_state = run_claim_pipeline(case["raw_document_text"])
        extraction = final_state["extraction"]
        decision = final_state["decision"]

        correct, total, detail = _field_match(
            case["expected_extraction"], extraction.model_dump(mode="json")
        )
        total_field_correct += correct
        total_fields += total

        status_match = decision.status.value == case["expected_status"]
        if status_match:
            decision_correct += 1

        results.append(
            {
                "id": case["id"],
                "field_accuracy": f"{correct}/{total}",
                "field_detail": detail,
                "expected_status": case["expected_status"],
                "actual_status": decision.status.value,
                "status_match": status_match,
            }
        )

    return {
        "num_cases": len(cases),
        "extraction_field_accuracy": round(total_field_correct / total_fields, 3) if total_fields else 0,
        "decision_accuracy": round(decision_correct / len(cases), 3) if cases else 0,
        "results": results,
    }


if __name__ == "__main__":
    report = run_eval()
    print(json.dumps(report, indent=2))
    # Fail CI if quality drops below threshold
    assert report["decision_accuracy"] >= 0.7, "Decision accuracy below threshold!"
    assert report["extraction_field_accuracy"] >= 0.7, "Extraction accuracy below threshold!"
