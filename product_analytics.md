Product analytics for RAG platform
=================================

Overview
--------
This guide explains what product analytics is for the Retrieval-Augmented Generation (RAG) platform, why it matters, how to implement it, and trade-offs to consider. The goal is to turn telemetry and user interactions into actionable product decisions: improve relevance, reduce latency, increase adoption, and detect failures.

Goals
-----
- Measure usage and adoption (who uses the system, frequency, retention).
- Measure product quality (answer accuracy, hallucination rate, source use).
- Measure performance and reliability (latency, errors, ingestion freshness).
- Measure business impact (time-to-answer, user satisfaction, cost per query).
- Support experiments and product decisions (A/B tests, rollout metrics).

Key product metrics (what & why)
-------------------------------
- Active Users (DAU/WAU/MAU): adoption and engagement.
- Queries per user (usage intensity): discover power users and typical load.
- Retention / Stickiness: whether users return after first use.
- Task Success / FCR (First Contact Resolution): percentage of queries resolved without escalation.
- CSAT / Rating: direct user feedback (1–5) after a query.
- Time-to-Answer: user-perceived speed; affects satisfaction.
- Query Success Rate (business-defined): % answers judged correct or useful.
- Hallucination Rate: % answers unsupported by returned sources.
- Escalation Rate: % queries that require human follow-up.
- Cost per 1k queries: operational cost metric for pricing and efficiency.

Quality & evaluation metrics (tie to `evals/`)
---------------------------------------------
- Precision@1, Precision@k, MRR: retrieval quality mapped to user success.
- Answer accuracy: human-labeled or automated checks on golden queries.
- Attribution coverage: whether answers include sources (trust metric).

Event schema and instrumentation (how)
-------------------------------------
Instrument structured events at these boundaries: client UI, API gateway, agent orchestration, ingestion pipeline, and guardrails. Include correlated ids for tracing: `user_id`, `session_id`, `query_id`, `trace_id`, `ingest_job_id`.

Core events (suggested)
- `query_submitted` { user_id, session_id, query_id, query_text, timestamp, client, route, user_tier }
- `retrieval_result` { query_id, retrieval_id, rank, doc_id, score, embeddings_model, vector_store, timestamp }
- `answer_returned` { query_id, answer_id, answer_text_length, sources:[doc_id], model, model_latency_ms, total_latency_ms }
- `user_feedback` { query_id, answer_id, rating, feedback_text, timestamp }
- `answer_clicked` { query_id, answer_id, doc_id, click_pos, timestamp }
- `escalation` { query_id, reason, target_team, timestamp }
- `ingest_started` / `ingest_completed` / `ingest_failed` { ingest_job_id, doc_id, size_bytes, chunks, timestamp }
- `guardrail_action` { query_id, rule_id, action, reason, timestamp }

Event payload best practices
- Use concise, typed JSON fields; avoid large blobs in events.
- Include timestamps in ISO8601 and monotonic durations in ms.
- Add version tags for schema evolution (`schema_v`).
- Emit enriched context (embedding model, index version, vector-store revision) to reproduce results.

Data pipeline (how to collect & store)
------------------------------------
1. Collect events from clients and services via a streaming layer (Segment, Azure Event Hubs, or Service Bus).
2. Persist raw events to a cold store (Blob Storage / Data Lake) for replay and audits.
3. Stream events into a warehouse (Synapse, Snowflake, or Postgres + Airbyte) for analytics and dashboards.
4. Run ETL jobs to compute aggregates and KPIs (dbt, Azure Data Factory, or scheduled Python jobs).
5. Visualize KPIs in dashboards (Power BI, Grafana, Metabase).

Dashboards & reports (what to show)
----------------------------------
- Overview: DAU/WAU/MAU, queries/day, QPS, cost per 1k queries.
- Quality: P@1, MRR (on golden queries), sampled hallucination rate, CSAT.
- Performance: P50/P95 latency, ingestion throughput, embedding latency.
- Safety: guardrail block rate, PII incidents, escalation rate.
- Cohorts: new vs returning users, by customer tier, by docs indexed.
- Funnels: discovery → first query → positive feedback → retention.

Experimentation & A/B testing (how)
----------------------------------
- Use feature flags or routing to assign a percentage of traffic to treatments.
- Track treatment assignment per `query_id` and include in events.
- Compute primary metrics (conversion, accuracy, latency) with confidence intervals; use sequential testing or Bayesian methods for production experiments.
- Guard against bias: randomize by `user_id` or `session_id`, not by request.

Privacy, compliance, and sampling (why / why not)
-----------------------------------------------
- Collect only necessary fields; redaction of PII at ingestion; store raw text encrypted and access-controlled.
- Sampling: for very high QPS, sample events or store summary metrics to reduce cost — but sampling reduces ability to debug individual incidents.
- Why not collect everything: cost, privacy risk, and regulatory exposure.

Trade-offs and guidance
----------------------
- Raw events vs aggregated metrics: raw events enable deep debugging and replay but are expensive; keep raw for short retention, aggregate for long-term.
- High-fidelity tracing vs cost: distributed tracing (OpenTelemetry) helps debug latency sources but adds overhead; sample traces for production (e.g., 1% or error-only sampling).
- Synchronous analytics vs async processing: synchronous computation (real-time) gives immediate insight but increases latency; prefer async for heavy aggregations.

Implementation steps (practical)
------------------------------
1. Define core events and implement instrumentation in `gateway/client.py` and `agents/graph.py` for `query_submitted`, `retrieval_result`, and `answer_returned`.
2. Pipe events to Azure Event Hubs / Service Bus and persist raw events to Blob Storage.
3. Add a nightly ETL to compute golden-query KPIs using `evals/` and push results to a dashboard.
4. Build Power BI / Grafana dashboards for the KPIs above and set SLO/alerts in Application Insights.
5. Add sampling and schema versioning; add retention policies for raw events.

Concrete examples & quick queries
--------------------------------
- Count queries per user:
  SELECT user_id, COUNT(*) AS q_count
  FROM events
  WHERE event_type='query_submitted' AND timestamp BETWEEN '{{start}}' AND '{{end}}'
  GROUP BY user_id
  ORDER BY q_count DESC;

- Weekly hallucination sample rate (requires human labels table `labels`):
  SELECT COUNT(*) FILTER (WHERE label='hallucination')::float / COUNT(*) AS hallucination_rate
  FROM labels
  WHERE label_date BETWEEN '{{start}}' AND '{{end}}';

Tooling recommendations
----------------------
- Event collection: Azure Event Hubs or Segment.
- Storage & warehouse: Azure Data Lake / Synapse or Snowflake for larger scale.
- BI: Power BI for business stakeholders; Grafana for real-time infra KPIs.
- Experimentation: LaunchDarkly / internal flags + DB-backed metric collection.
- Tracing: OpenTelemetry + Application Insights (sampled traces).

Why product analytics matters (recap)
-----------------------------------
- Converts telemetry into product decisions: what to improve (retrieval, UX, costs) and whether changes help customers.
- Identifies regressions early via automated regression runs + dashboards.
- Enables data-driven experiments and prioritization.

Why not (risks / when to skip)
-----------------------------
- Small prototype teams: heavy analytics stack may be overkill early on; prefer simple logs and targeted surveys.
- High privacy/regulatory constraints: may need limited telemetry and strict retention, reducing insight.

Next steps I can take
---------------------
- Instrument `gateway/client.py` and `agents/graph.py` with `query_submitted` and `answer_returned` events.
- Scaffold ETL that produces nightly KPI reports (`out/product_analytics_report.json`).
- Create a starter Power BI/Grafana dashboard template.

Appendix: mapping to repo files
------------------------------
- Instrument these files first: `gateway/client.py`, `agents/graph.py`, `ingestion/processor.py`, `guardrails/rails.py`, `evals/run_evals.py`.
