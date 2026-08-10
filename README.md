# InsureAgent-RAG
### Autonomous P&C Insurance Claims Copilot — Multi-Agent RAG System

An AI-powered claims processing assistant that extracts structured data from unstructured
insurance documents, validates claims against policy rules using RAG, flags missing
information, and produces a structured decision recommendation for human adjusters.

Built to demonstrate the full AI Prompt Engineer + Platform Engineer skill set: agentic
workflows, RAG, structured outputs, evaluation frameworks, API design, containerization,
CI/CD, and observability.

**Status:** eval suite passing at 100% extraction accuracy / 100% decision accuracy across
5 labeled cases spanning two policy domains (homeowners, auto) and all four decision
outcomes. Average request latency ~4s end-to-end (Groq extraction + local embeddings +
Neon/pgvector retrieval + decision synthesis).

---

## 1. Problem It Solves

Insurance adjusters manually read claim notes, First Notice of Loss (FNOL) forms, and
policy booklets to decide whether a claim is valid, needs more info, or should be flagged
for fraud review. This project automates the first-pass triage using a 3-agent pipeline
grounded in real policy documents (not hallucinated answers).

## 2. Architecture

```
                     ┌─────────────────────┐
   Raw Claim Doc ───▶│  Extraction Agent    │──▶ Structured JSON (Pydantic-validated)
   (PDF / text)       │  (LLM + prompt)      │       │
                     └─────────────────────┘       │
                                                     ▼
                     ┌─────────────────────┐   Claim Data
                     │  Policy Compliance   │◀──────┘
                     │  Agent (RAG)         │
                     │  - pgVector / Chroma │
                     └─────────────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │  Decision Agent      │──▶ Final structured recommendation
                     │  (synthesis)         │    {status, confidence, missing_fields,
                     └─────────────────────┘     reasoning, citations}
                                │
                                ▼
                     FastAPI  /process-claim
                                │
                                ▼
                     PostgreSQL (audit log / metadata)
                                │
                                ▼
                     Monitoring: structured logs + /metrics
```

## 3. Core Components

| Layer | Tech | Maps to JD skill |
|---|---|---|
| Extraction Agent | Groq (Llama 3.x) + Pydantic | Structured Output Design, Prompt Engineering |
| Policy Compliance Agent | LangGraph/CrewAI + pgVector/Chroma | RAG Architecture, Agentic AI, Vector DBs |
| Decision Agent | LLM synthesis + rules | Agentic AI, LLM reasoning |
| API | FastAPI | REST APIs, FastAPI |
| Storage | PostgreSQL | PostgreSQL, Data pipelines |
| Eval | Custom scoring harness | LLM Evaluation Frameworks |
| Deployment | Docker + GitHub Actions | CI/CD, DevOps |
| Observability | Structured logging + `/metrics` | Observability & Monitoring |

## 3b. Orchestration (LangGraph)

The three agents are wired together as an explicit `StateGraph` in `app/graph.py`
rather than a plain sequential function chain:

```
START -> extract_node -> [route_after_extract]
                              |-- "compliance"      --> compliance_node --> decide_node --> END
                              |-- "needs_more_info"  --> needs_more_info_node --> END
```

The conditional edge (`route_after_extract`) inspects the Extraction Agent's output:
if required fields (`claim_id`, `policyholder_name`, `policy_number`, `loss_description`)
are missing, the graph short-circuits straight to a `MORE_INFO_NEEDED` decision and
**skips the RAG/compliance call entirely** — saving an LLM call and a vector-store
round trip on incomplete submissions. Both the API (`app/main.py`) and the eval harness
(`app/eval/run_eval.py`) invoke the same compiled graph, so evaluation always reflects
the real production path, not a hand-rolled shortcut.

This structure also makes the pipeline easy to extend later — e.g. adding a loop-back
edge from `needs_more_info_node` to `extract_node` for a re-prompt/human-in-the-loop
flow, or a parallel branch that runs a fraud-signal agent alongside compliance.

## 4. Agent Responsibilities

**Agent 1 — Extraction Agent**
Reads raw claim text/PDF → outputs strict JSON: `claim_id, policyholder_name, loss_date,
claim_amount, loss_description, policy_number`. Validated against a Pydantic schema; retries
with a repair prompt if validation fails.

**Agent 2 — Policy Compliance (RAG) Agent**
Embeds the claim's key facts, retrieves relevant policy clauses/deductibles from the vector
store, and determines coverage alignment with citations back to the source policy text.
`data/policy_docs/` ships with two independent policy documents (homeowners and auto), so
retrieval has to correctly discriminate between domains rather than just returning
whatever's in the store — a water-damage claim retrieves homeowners clauses, a collision
claim retrieves auto clauses, without the two ever cross-contaminating.

**Agent 3 — Decision Agent**
Combines Agent 1 + Agent 2 outputs into a final recommendation: `APPROVED`,
`FLAGGED_FOR_FRAUD`, or `MORE_INFO_NEEDED`, with a confidence score and reasoning.

## 5. API Endpoints

- `POST /process-claim` — full pipeline, input: raw document text, output: structured decision
- `GET /claims/{claim_id}` — fetch a prior decision from PostgreSQL
- `GET /metrics` — basic observability (latency, token usage, request count)
- `GET /eval-report` — runs the eval suite and returns accuracy scores

## 6. Evaluation Framework

`app/eval/eval_set.json` contains labeled synthetic claims spanning both policy domains
(homeowners + auto) and all four decision outcomes (`APPROVED`, `MORE_INFO_NEEDED`,
`FLAGGED_FOR_FRAUD`, and cases exercising the graph's short-circuit path). `run_eval.py`
scores two things:
- **Extraction accuracy** — critical fields (IDs, amounts, dates, names) are scored with
  exact match; descriptive fields (`loss_type`, summaries) use fuzzy/containment matching,
  since a correct-but-differently-phrased answer shouldn't count as a failure.
- **Decision accuracy** — whether the final `status` matches the expected label.

Runs in CI on every push (`.github/workflows/ci.yml`) against a disposable
`pgvector/pgvector:pg16` container — not the real Neon database — so prompt changes can't
silently regress quality, and CI never touches production data. `python -m app.eval.run_eval`
exits non-zero if either score drops below 70%, failing the build.

## 7. Setup

**Database:** this project uses [Neon](https://neon.tech) (serverless Postgres with pgvector
built in) rather than a local/Dockerized Postgres — no native compiler, no Docker daemon
required. Create a free Neon project, run `CREATE EXTENSION IF NOT EXISTS vector;` once in
Neon's SQL Editor, then copy the connection string into `.env` as `DATABASE_URL`.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your Groq API key + Neon connection string
python -m app.rag.ingest  # embed sample policy docs (run once)
uvicorn app.main:app --reload          # terminal 1 — API on :8000
streamlit run streamlit_app.py         # terminal 2 — UI on :8501
```

Or via Docker (API + Streamlit only — Neon replaces the local Postgres container):
```bash
docker compose up -d
```

## 7b. Streamlit Frontend

`streamlit_app.py` is a thin UI over the FastAPI backend — no business logic lives here,
it's purely a client, so the API stays the real product surface (testable, scriptable,
usable by other systems) while the UI just makes it demoable.

Pages:
- **Process a Claim** — paste raw claim text, see live output from all three agents
  (extraction fields, coverage determination + citations, final decision with confidence)
- **Look Up Past Claim** — pull a prior decision from Postgres by claim ID
- **Eval Report** — trigger the eval suite and see per-case accuracy in the browser
- **Live Metrics** — request count, error rate, avg/p95 latency from `/metrics`

It talks to the API via `API_BASE_URL` (defaults to `http://localhost:8000`; in Docker
Compose this is set to `http://api:8000` automatically since containers reach each other
by service name, not `localhost`).

Once running: open `http://localhost:8501`.

## 8. What Was Actually Built

Beyond the core pipeline, a few things came out of real debugging along the way that are
worth calling out on their own:

- **LangGraph conditional routing** — the graph short-circuits to `MORE_INFO_NEEDED` and
  skips the RAG call entirely when required fields are missing, rather than always running
  all three agents regardless of input quality.
- **Fraud-vs-denial ordering bug** — the Decision Agent originally checked "is it covered"
  before "does it look fraudulent," so a suspicious-but-technically-uncovered claim got
  silently `DENIED` instead of flagged for human review. Caught by the eval suite, fixed by
  reordering the decision rules so fraud signals are checked first.
- **Extraction over-conservatism bug** — the Extraction Agent initially refused to infer
  `loss_type` from context (e.g. "burst pipe flooded the floor" → left `loss_type: null`
  instead of inferring "water damage"), which cascaded into an incorrect `MORE_INFO_NEEDED`.
  Fixed by explicitly distinguishing factual fields (never guess) from categorical fields
  (safe to infer) in the prompt.
- **Eval scorer type-coercion bug** — `claim_amount: 8500` vs `8500.0` was being scored as
  a mismatch due to naive string comparison, making a fully correct extraction look like an
  84%-accuracy system. Fixed by normalizing numeric types before comparison, with a
  regression test locking in the fix.
- **Neon connection pooling** — the original implementation opened a fresh Postgres
  connection per RAG lookup and per DB write, which is expensive against serverless
  compute (SSL handshake + potential cold-start). Adding a shared `ThreadedConnectionPool`
  cut end-to-end request latency from ~26s to ~4s.

## 9. Build Order 

1. Pydantic schemas + Extraction Agent + unit tests
2. Ingested sample policy docs (homeowners, then auto) → local embeddings → pgVector on Neon
3. Policy Compliance RAG Agent, with citation tracing back to source clauses
4. Decision Agent + LangGraph orchestration wiring all 3 agents with conditional routing
5. FastAPI endpoints + PostgreSQL logging
6. Eval harness + labeled eval set, iterated as real bugs surfaced
7. Streamlit frontend as a thin client over the API
8. Dockerfile + docker-compose + GitHub Actions CI
9. Swapped OpenAI → Groq (LLM) + local sentence-transformers (embeddings) to remove paid
   API dependencies entirely; migrated local/Docker Postgres → Neon for zero-setup RAG storage
10. Connection pooling + eval scorer fixes, driven by real accuracy/latency numbers from
    the running system, not assumptions

## 10. Pushing to GitHub (with CI actually running)

```bash
git init
git add .
git commit -m "Initial commit: InsureAgent-RAG"
git branch -M main
git remote add origin https://github.com/<your-username>/InsureAgent-RAG.git
git push -u origin main
```

The CI workflow (`.github/workflows/ci.yml`) needs one repo secret to actually run:

1. On GitHub, go to your repo → **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Name: `GROQ_API_KEY`, Value: your real Groq key
4. Save

CI spins up its own disposable `pgvector/pgvector:pg16` container for the eval run — it
never touches your real Neon database, so pushing won't affect production data. Once the
secret is set, every push to `main` (and every PR) will automatically: install
dependencies, run unit tests, run the full eval suite, and build the Docker image — failing
the build if extraction or decision accuracy drops below 70%.

Check the **Actions** tab on GitHub after pushing to watch it run.
  
## Author 
```
Toran V Athani
```
