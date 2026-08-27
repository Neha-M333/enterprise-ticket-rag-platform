# Design Note — Enterprise Ticket RAG Platform

## 1. Production Failure Modes

### 1.1 LLM API Rate Limits / Outages

**Failure.** The Gemini API returns `429 RESOURCE_EXHAUSTED` (quota exceeded) or a
transient `5xx`. This is not hypothetical — it happened repeatedly during
development of `eval_harness.py` on a free-tier key capped at 20
requests/day for `gemini-3.6-flash`, and a second key was needed to get a
clean run.

**Detection.** Every call to `triage_ticket()` / `evaluate_account_health()`
runs through a wrapper (`_call_with_backoff` in the eval harness) that
inspects the exception string for `RESOURCE_EXHAUSTED` / `429`. Critically,
it distinguishes a **per-minute** rate limit (worth a short retry) from a
**per-day** quota (retrying within the same run cannot help, since the
`retryDelay` Gemini returns for a daily cap is a generic backoff hint, not a
countdown to reset).

**Mitigation.** Three layers: (1) response caching so semantically
identical work isn't repeated — the harness triages a given `ticket_id` or
evaluates a given `account_id` at most once per run, even if multiple test
cases inspect different facets of the same result; (2) bounded exponential
backoff for genuinely transient errors; (3) fail gracefully rather than
crash — a quota-exhausted test is marked `SKIPPED`, not `FAIL`, so a
capacity problem is never conflated with a code defect in the report. In a
live production system this would extend to a request queue with
backpressure, and a fallback path (e.g., rules-based priority defaulting to
"Medium" pending human triage) so ticket ingestion never blocks on the LLM
being available.

### 1.2 Retrieval Hallucination / Out-of-Domain Query

**Failure.** A ticket's topic isn't covered by the knowledge base (only
three articles exist: `analyticshub.md`, `authentication-sso.md`,
`workflowengine.md`). The retriever still returns its top-k KB IDs by rank
even when true relevance is near zero, and an ungrounded LLM can then
fabricate a confident-sounding but unsupported `draft_response`.

**Detection.** `KBRetriever.retrieve()` exposes a numeric similarity score
per hit (cosine similarity under TF-IDF, or the keyword-overlap ratio in
the pure-Python fallback). A near-zero top score across all candidates is
a reliable out-of-domain signal.

**Mitigation.** Add a similarity threshold: below it, the prompt should
state explicitly that no relevant KB content was found (the harness already
does this — `"No relevant knowledge base articles were found."`) rather
than passing borderline-irrelevant snippets as if they were authoritative.
The `retrieved_kb_ids` field in `TicketTriageResponse` should be empty in
that case, which becomes a queryable signal for "expand the knowledge
base here" — a genuine KB gap turns into a backlog item instead of a
silent wrong answer to a customer.

### 1.3 Schema Parsing Failure

**Failure.** Despite `response_mime_type: "application/json"` and a
`response_schema` pinned to the Pydantic model, the model can still return
JSON that fails schema validation (missing field, wrong enum value, or the
SDK's `.parsed` attribute not populating on an older SDK version).

**Detection.** `triage_ticket()` and `evaluate_account_health()` both check
`response.parsed` first and fall back to `json.loads(response.text)` +
manual `TicketTriageResponse(**data)` / `AccountHealthResponse(**data)`
construction, which raises a `pydantic.ValidationError` on any mismatch.
`eval_harness.py`'s Test 4 (Deterministic Schema Round-Trip) exists
specifically to catch regressions here: it serializes a live result back to
JSON and re-parses it, asserting the two are identical.

**Mitigation.** Structured output constrained by `response_schema` is the
primary defense. As a second layer, a validation failure should trigger one
retry with a corrective instruction appended to the prompt ("your previous
response did not match the required schema — respond with valid JSON
only") before falling back to a safe default (e.g., `priority="Medium"`,
flagged for human review) rather than raising an unhandled exception into
the ticketing pipeline.

## 2. Latency vs. Quality Trade-offs

The retrieval layer uses TF-IDF over a small, static markdown corpus rather
than a dense vector database (Pinecone, FAISS + embeddings). At three KB
documents, cosine similarity over sparse TF-IDF vectors and a dense
embedding index return functionally equivalent rankings, but TF-IDF needs
no embedding API call, no vector store infrastructure, and re-indexes
in-memory in milliseconds — the entire retrieval step adds negligible
latency ahead of the LLM call, which dominates end-to-end time regardless.
A pure-Python keyword-overlap fallback exists for environments without
`scikit-learn`, trading a small amount of ranking quality for zero
dependency risk.

On the generation side, `gemini-3.6-flash` was chosen over a larger model
or a locally hosted LLM specifically because ticket triage is a
low-complexity, high-volume classification task — the schema is narrow
(4 priority values, 4 categories, a short draft), not open-ended
reasoning. A larger model would add cost and latency without materially
improving classification accuracy on a task this constrained. A local
model would remove API dependency risk (Section 1.1) but introduces GPU
provisioning cost and ongoing maintenance that isn't justified until
request volume is high enough to amortize it (see Section 4).

## 3. Data Sensitivity & PII

Account records contain a `primary_contact` (name, title) and free-text
`escalation_notes`, both of which are sent into LLM prompts in
`task2_account.py`'s `_build_prompt()`. Before this reaches a remote
endpoint in production, three techniques apply: (1) a regex-based scrub
pass over free-text fields (ticket `description`, `escalation_notes`) for
emails, phone numbers, and payment-card-like digit sequences, replacing
matches with typed placeholder tokens (`[EMAIL]`, `[PHONE]`) so the model
still sees *that* PII existed without seeing its value; (2) substituting
`account_id` for `company`/`primary_contact.name` in the prompt wherever
the LLM doesn't need the real identity to do its job — the current code
already keeps `company` and `account_id` separate, so this is a targeted
change rather than a redesign; (3) routing any field explicitly flagged as
sensitive to a separate, unlogged prompt-construction path so redaction
failures are auditable independent of the main pipeline.

## 4. Scaling Considerations (500 → 50,000 tickets)

At 500 tickets, `parse_data.py` loading everything from flat
`tickets_batch_*.json` files and re-fitting `TfidfVectorizer` on every
`KBRetriever()` instantiation is fine. At 50,000, three bottlenecks
emerge. **Indexing:** flat-file glob loading doesn't scale — tickets need
to live in a real datastore (Postgres/SQLite at minimum) with the 90-day
window query pushed into SQL rather than pandas filtering the entire table
in memory on every call. **Chunking/retrieval:** once the knowledge base
itself grows past a few hundred articles, TF-IDF's advantage over embeddings
narrows, and a proper vector index (pgvector or FAISS) with periodic
re-indexing (not on every process start) becomes worthwhile. **API
throughput:** sequential `generate_content()` calls per ticket become the
real bottleneck — Gemini's batch API (async batch job submission instead
of one request per ticket) and response caching by content hash (identical
or near-identical tickets shouldn't re-trigger a fresh LLM call) are the
two highest-leverage changes, alongside moving the daily-quota-aware retry
logic already in `eval_harness.py` into the production triage path itself.
