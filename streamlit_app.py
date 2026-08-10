"""
InsureAgent-RAG — Streamlit frontend.

A thin UI over the FastAPI backend: submit a raw claim document, see the
Extraction / Compliance / Decision agent outputs, browse past claims, and
check the eval report + live metrics.

Run:
    streamlit run streamlit_app.py

Assumes the FastAPI backend is already running (see README) at API_BASE_URL.
"""
import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="InsureAgent-RAG", page_icon="📋", layout="wide")

SAMPLE_CLAIM = (
    "FNOL Form. Claim ID: CLM-2044. Policyholder: Anita Desai. "
    "Policy Number: PCY-77410. Date of Loss: 2026-07-20. "
    "Description: Burst pipe under kitchen sink flooded the lower floor overnight. "
    "Estimated claim amount: $6,200."
)

STATUS_COLORS = {
    "APPROVED": "green",
    "MORE_INFO_NEEDED": "orange",
    "FLAGGED_FOR_FRAUD": "red",
    "DENIED": "red",
}


def api_healthy() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def render_extraction(extraction: dict):
    st.subheader("🔍 Extraction Agent")
    cols = st.columns(3)
    cols[0].metric("Claim ID", extraction.get("claim_id") or "—")
    cols[1].metric("Policyholder", extraction.get("policyholder_name") or "—")
    cols[2].metric("Claim Amount", f"${extraction['claim_amount']:,}" if extraction.get("claim_amount") else "—")

    cols2 = st.columns(3)
    cols2[0].write(f"**Policy #:** {extraction.get('policy_number') or '—'}")
    cols2[1].write(f"**Loss Date:** {extraction.get('loss_date') or '—'}")
    cols2[2].write(f"**Loss Type:** {extraction.get('loss_type') or '—'}")

    st.write(f"**Description:** {extraction.get('loss_description') or '—'}")

    if extraction.get("missing_fields"):
        st.warning(f"Missing fields: {', '.join(extraction['missing_fields'])}")


def render_compliance(compliance: dict):
    st.subheader("📖 Policy Compliance Agent (RAG)")
    covered = compliance.get("is_covered")
    label = "✅ Covered" if covered is True else "❌ Not Covered" if covered is False else "❓ Inconclusive"
    st.write(f"**Coverage:** {label}")

    cols = st.columns(2)
    deductible_display = f"${compliance['deductible']:,}" if compliance.get("deductible") else "—"
    limit_display = f"${compliance['coverage_limit']:,}" if compliance.get("coverage_limit") else "—"
    cols[0].write(f"**Deductible:** {deductible_display}")
    cols[1].write(f"**Coverage Limit:** {limit_display}")

    if compliance.get("notes"):
        st.caption(compliance["notes"])

    citations = compliance.get("citations") or []
    if citations:
        with st.expander(f"📎 {len(citations)} policy citation(s)"):
            for c in citations:
                st.markdown(f"**{c['source']}**")
                st.write(c["clause_text"])
                st.divider()


def render_decision(decision: dict):
    st.subheader("⚖️ Decision Agent")
    status = decision.get("status", "UNKNOWN")
    color = STATUS_COLORS.get(status, "gray")
    st.markdown(f"### :{color}[{status}]")
    st.progress(decision.get("confidence", 0), text=f"Confidence: {decision.get('confidence', 0):.0%}")
    st.write(decision.get("reasoning", ""))

    if decision.get("missing_fields"):
        st.warning(f"Missing fields: {', '.join(decision['missing_fields'])}")


# ---------- Sidebar ----------

with st.sidebar:
    st.title("📋 InsureAgent-RAG")
    st.caption("Autonomous P&C Insurance Claims Copilot")

    if api_healthy():
        st.success(f"Connected to API\n\n{API_BASE_URL}")
    else:
        st.error(f"Cannot reach API at {API_BASE_URL}\n\nStart it with:\n`uvicorn app.main:app --reload`")

    st.divider()
    page = st.radio("Navigate", ["Process a Claim", "Look Up Past Claim", "Eval Report", "Live Metrics"])

# ---------- Pages ----------

if page == "Process a Claim":
    st.header("Process a New Claim")
    st.caption("Paste raw claim text (FNOL form, adjuster notes, etc.) and run it through the 3-agent pipeline.")

    raw_text = st.text_area("Raw claim document text", value=SAMPLE_CLAIM, height=150)

    if st.button("🚀 Process Claim", type="primary"):
        with st.spinner("Running Extraction → Compliance → Decision agents..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/process-claim",
                    json={"raw_document_text": raw_text},
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()

                render_extraction(result["extraction"])
                st.divider()
                render_compliance(result["compliance"])
                st.divider()
                render_decision(result["decision"])

                with st.expander("🔧 Raw JSON response"):
                    st.json(result)

            except requests.HTTPError as e:
                st.error(f"API returned an error: {e.response.status_code} — {e.response.text}")
            except requests.RequestException as e:
                st.error(f"Could not reach the API: {e}")

elif page == "Look Up Past Claim":
    st.header("Look Up a Past Claim")
    claim_id = st.text_input("Claim ID", placeholder="e.g. CLM-2044")

    if st.button("🔎 Look Up") and claim_id:
        try:
            resp = requests.get(f"{API_BASE_URL}/claims/{claim_id}", timeout=10)
            if resp.status_code == 404:
                st.warning("No record found for that claim ID.")
            else:
                resp.raise_for_status()
                result = resp.json()
                st.write(f"**Status:** {result['status']}  |  **Confidence:** {result['confidence']:.0%}")
                st.caption(f"Processed at {result['created_at']}")
                st.json(result)
        except requests.RequestException as e:
            st.error(f"Could not reach the API: {e}")

elif page == "Eval Report":
    st.header("Prompt Evaluation Report")
    st.caption("Runs the labeled eval set against the live pipeline and scores extraction + decision accuracy.")

    if st.button("▶️ Run Eval Suite"):
        with st.spinner("Running eval cases..."):
            try:
                resp = requests.get(f"{API_BASE_URL}/eval-report", timeout=120)
                resp.raise_for_status()
                report = resp.json()

                cols = st.columns(3)
                cols[0].metric("Cases", report["num_cases"])
                cols[1].metric("Extraction Accuracy", f"{report['extraction_field_accuracy']:.0%}")
                cols[2].metric("Decision Accuracy", f"{report['decision_accuracy']:.0%}")

                st.divider()
                for r in report["results"]:
                    icon = "✅" if r["status_match"] else "❌"
                    st.write(
                        f"{icon} **{r['id']}** — fields: {r['field_accuracy']}, "
                        f"expected: `{r['expected_status']}`, got: `{r['actual_status']}`"
                    )
                    field_detail = r.get("field_detail") or []
                    mismatches = [f for f in field_detail if not f["match"]]
                    if mismatches:
                        with st.expander(f"⚠️ {len(mismatches)} field mismatch(es) in {r['id']}"):
                            for f in mismatches:
                                st.write(
                                    f"**{f['field']}** ({f['method']}) — "
                                    f"expected: `{f['expected']}`, got: `{f['actual']}`"
                                )
            except requests.RequestException as e:
                st.error(f"Could not reach the API: {e}")

elif page == "Live Metrics":
    st.header("Live API Metrics")
    if st.button("🔄 Refresh"):
        pass  # button click alone triggers rerun

    try:
        resp = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        resp.raise_for_status()
        m = resp.json()

        cols = st.columns(4)
        cols[0].metric("Total Requests", m["total_requests"])
        cols[1].metric("Error Rate", f"{m['error_rate']:.1%}")
        cols[2].metric("Avg Latency", f"{m['avg_latency_ms']:.0f} ms")
        cols[3].metric("p95 Latency", f"{m['p95_latency_ms']:.0f} ms")

        st.caption(f"Sample size: {m['sample_size']} recent requests")
    except requests.RequestException as e:
        st.error(f"Could not reach the API: {e}")
