"""
LangGraph orchestration for the InsureAgent-RAG pipeline.

Wires the three agents into an explicit state graph instead of a straight
sequential function-call chain. This makes the workflow inspectable,
resumable, and easy to extend (e.g. add a conditional branch that routes
back to the Extraction Agent if required fields are missing, instead of
plowing through to a decision on incomplete data).

Graph shape:

    START -> extract -> route_after_extract -> (needs_more_info | compliance)
    compliance -> decide -> END
    needs_more_info -> END
"""
import logging
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.compliance_agent import check_compliance
from app.agents.decision_agent import make_decision
from app.agents.extraction_agent import extract_claim
from app.models.schemas import (
    ClaimDecision,
    ClaimStatus,
    ComplianceResult,
    ExtractedClaim,
)

logger = logging.getLogger(__name__)

# Fields that MUST be present before it's worth spending a compliance/RAG call.
REQUIRED_FIELDS = ["claim_id", "policyholder_name", "policy_number", "loss_description"]


class ClaimState(TypedDict, total=False):
    raw_document_text: str
    extraction: ExtractedClaim
    compliance: ComplianceResult
    decision: ClaimDecision


# ---------- Node functions ----------

def extract_node(state: ClaimState) -> ClaimState:
    logger.info("[graph] running extraction_agent")
    extraction = extract_claim(state["raw_document_text"])
    return {"extraction": extraction}


def compliance_node(state: ClaimState) -> ClaimState:
    logger.info("[graph] running compliance_agent")
    compliance = check_compliance(state["extraction"])
    return {"compliance": compliance}


def decision_node(state: ClaimState) -> ClaimState:
    logger.info("[graph] running decision_agent")
    decision = make_decision(state["extraction"], state["compliance"])
    return {"decision": decision}


def needs_more_info_node(state: ClaimState) -> ClaimState:
    """
    Short-circuit path: extraction is missing required fields, so we skip
    the (expensive) RAG compliance call entirely and return a decision
    directly, citing which fields are missing.
    """
    logger.info("[graph] short-circuit: required fields missing, skipping compliance agent")
    extraction = state["extraction"]
    decision = ClaimDecision(
        claim_id=extraction.claim_id or "UNKNOWN",
        status=ClaimStatus.MORE_INFO_NEEDED,
        confidence=0.95,
        reasoning=(
            "Required fields are missing from the submitted document, so a coverage "
            "determination cannot be made yet."
        ),
        missing_fields=extraction.missing_fields,
        citations=[],
    )
    return {"compliance": ComplianceResult(), "decision": decision}


# ---------- Routing ----------

def route_after_extract(state: ClaimState) -> str:
    extraction = state["extraction"]
    missing_required = [
        f for f in REQUIRED_FIELDS
        if not getattr(extraction, f, None) or f in extraction.missing_fields
    ]
    if missing_required:
        return "needs_more_info"
    return "compliance"


# ---------- Graph assembly ----------

def build_graph():
    graph = StateGraph(ClaimState)

    graph.add_node("extract_node", extract_node)
    graph.add_node("compliance_node", compliance_node)
    graph.add_node("decide_node", decision_node)
    graph.add_node("needs_more_info_node", needs_more_info_node)

    graph.set_entry_point("extract_node")

    graph.add_conditional_edges(
        "extract_node",
        route_after_extract,
        {
            "compliance": "compliance_node",
            "needs_more_info": "needs_more_info_node",
        },
    )

    graph.add_edge("compliance_node", "decide_node")
    graph.add_edge("decide_node", END)
    graph.add_edge("needs_more_info_node", END)

    return graph.compile()


# Compiled once at import time and reused across requests.
claim_graph = build_graph()


def run_claim_pipeline(raw_document_text: str) -> ClaimState:
    """Entry point used by the FastAPI layer."""
    final_state = claim_graph.invoke({"raw_document_text": raw_document_text})
    return final_state
