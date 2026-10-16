Production deployment notes
==========================

Overview
--------
This document captures learnings, issues faced, and rationale for productionizing the Retrieval-Augmented Generation (RAG) platform used by medium-sized enterprise clients on Azure. It explains the deployment choices, scale considerations, operational practices, and alternatives considered.

Assumptions
-----------
- Target customers: medium-sized enterprises (10s–100s of internal users, moderate query volume)
- Cloud: Azure (VMs, App Service, Azure Functions, Azure Container Registry, AKS, Azure Cognitive Services if used)
- Data: mix of PDFs, docs, presentations, internal databases, and uploaded files
- SLA: best-effort low-latency responses for interactive queries; background ingestion pipelines may be eventual

Key components deployed
------------------------
- API / Gateway: exposes query endpoints and routes to agents; deployed as containerized web service (App Service or AKS)
- Ingestion workers: background services for crawling, extracting, chunking, and embedding documents (Azure Functions or container jobs)
- Vector store: hosted vector database (Azure Cosmos DB with vector extensions, or an external managed vector DB) for retrieval
- Embeddings: model inference either via managed service (Azure OpenAI / Cognitive Services embeddings) or self-hosted model on GPU VMs
- Agents & orchestration: stateless services that build prompts, apply guardrails, and call LLMs
- Logging/Monitoring: Azure Monitor, Application Insights, and centralized log storage
- Storage: Azure Blob Storage for raw files and preprocessed chunks

Learning steps and timeline
--------------------------
1. Local prototype (2–4 weeks)
   - Built end-to-end pipeline locally using small dataset; verified retrieval + prompt composition flow.
   - Validated chunk size, overlap, and embedding model choices.
2. Small cloud pilot (2–3 weeks)
   - Deployed basic components on Azure App Service + Blob Storage.
   - Used a managed embedding API to avoid GPU costs during iteration.
   - Observed cold-start and latency issues for large files during ingestion.
3. Scale & stability testing (3–4 weeks)
   - Simulated production ingestion volumes and query load using locust.
   - Identified bottlenecks: embedding rate limits, vector DB write throughput, and synchronous embedding latency.
4. Harden for production (4–6 weeks)
   - Introduced asynchronous ingestion, batching, retry policies, and backpressure.
   - Added guardrails and validation to prevent PII leakage and enforce business rules.
   - Implemented observability, health checks, and SLOs.

Operational issues faced and mitigations
--------------------------------------
- Embedding API rate limits & latency
  - Issue: managed embedding endpoints had throttling and variable latency affecting ingestion throughput.
  - Mitigation: add batching, exponential backoff retries, and a local queue (Azure Service Bus) to smooth bursts. For heavy clients, provision dedicated embedding capacity or use GPU VMs.

- Vector store write/read throughput
  - Issue: initial vector store choice hit write limits under concurrent ingestion.
  - Mitigation: switched to partitioned collections, batched writes, and tuned RU/s (for Cosmos DB) or scaled the managed vector DB tier.

- Cost vs latency trade-offs for embeddings
  - Issue: managed embeddings are simple but costly at scale; self-hosting reduces cost but increases operational complexity.
  - Mitigation: hybrid approach — use managed embeddings for cold-starts and critical queries, and self-hosted for bulk offline batch embedding.

- Model hallucinations and incorrect answers
  - Issue: LLMs may produce confident but incorrect answers.
  - Mitigation: strict retrieval-first design, answer attribution (return sources), guardrail validations, conservative prompt templates, and human-in-the-loop verification for sensitive domains.

- Data freshness and reindexing
  - Issue: keeping vector store in sync with changing documents.
  - Mitigation: change-data-capture (CDC) hooks, periodic reindex jobs, and a lightweight change-queue triggering incremental re-ingestion.

- Security and compliance
  - Issue: sensitive data handling and access control.
  - Mitigation: encrypt data at rest, use private networking (VNet), enforce role-based access, audit logs, and redact PII during ingestion.

Why Azure and why not alternatives
---------------------------------
- Why Azure
  - Integrated PaaS: App Service, Functions, Blob Storage, and Cosmos DB simplify many operational tasks and provide enterprise-grade SLAs.
  - Managed AI services: Azure OpenAI or Cognitive Services provide hosted embeddings and model APIs for rapid iteration.
  - Security: Azure AD, VNet integration, and compliance certifications match enterprise needs.
  - Observability: Azure Monitor and Application Insights provide integrated telemetry.

- Why not other clouds / self-hosting
  - AWS/GCP: comparable capabilities, but the team had existing Azure expertise and subscriptions, so Azure reduced onboarding friction.
  - Pure self-hosting: lowers per-transaction costs in some cases but increases ops burden (managing GPUs, scaling vector DBs, patching). For medium clients, Azure PaaS hits a good balance.

Production architecture choices and trade-offs
-------------------------------------------
- Containers vs Serverless
  - Chosen: containerized services for the API and agents (AKS or App Service) + serverless for small event-driven tasks.
  - Trade-off: containers provide consistent runtime and easier dependency management; serverless reduces management for small spikes but complicates cold-start sensitive paths.

- Managed vector DB vs Self-hosted
  - Chosen: managed vector store (Cosmos DB or managed vendor) for availability and lower ops.
  - Trade-off: managed services cost more, but reduce DB ops and provide scaling guarantees.

- Managed embeddings vs Self-hosted model
  - Chosen: hybrid. Use managed embeddings for low-latency, critical queries; use self-hosted batch GPU jobs for bulk processing.
  - Trade-off: hybrid reduces cost and risk while keeping fast paths.

Operational best practices
-------------------------
- Use asynchronous pipelines (Service Bus, durable functions) for ingestion and embedding to decouple rate limits and retries.
- Implement idempotent ingestion and strict deduplication.
- Enforce guardrails early in the pipeline and log reasons for blocked outputs.
- Instrument SLOs and alert on error budget, embedding latency, and vector-store saturation.
- Run periodic re-evaluation (golden queries) to detect drift and regressions.

Cost considerations (medium clients)
----------------------------------
- Primary costs: embedding API calls (managed), vector DB RU/s, storage, and model API usage.
- Optimization levers: batching embeddings, choosing lower-cost embedding models for non-critical data, caching frequent queries, and offloading bulk embedding to scheduled GPU jobs.

Next steps and scale-up guidance
-------------------------------
- For larger customers, consider:
  - Dedicated GPU inference fleet for low-latency LLM/embedding inference.
  - Multi-region vector DB replicas and read-routing.
  - Enhanced data governance workflows and customer-specific VNet peering.

Appendix: Useful links and references
------------------------------------
- See `docs/12_DEPLOYMENT.md` and `local_testing.md` for deployment scripts and runbooks.
- Key files:
  - `ingestion/processor.py` — ingestion orchestration
  - `agents/graph.py` — agent orchestration
  - `guardrails/rails.py` — policy enforcement

Practical metrics, explanations, and debugging guidance
-----------------------------------------------------

Sample practical metrics (assume production, medium client)
---------------------------------------------------------

| Metric | Sample value (medium client) | What it measures |
|---|---:|---|
| Precision@1 (P@1) | 0.70 | Fraction of queries where the top retrieved chunk leads to a correct answer |
| Precision@5 (P@5) | 0.85 | Fraction of the top 5 retrieved chunks containing at least one correct source |
| MRR | 0.75 | Mean Reciprocal Rank across golden queries; higher is better |
| Answer Accuracy | 0.82 | Fraction of final answers judged correct by human or auto checks |
| Hallucination rate | 6% | Fraction of answers judged unsupported by returned sources |
| Source attribution coverage | 98% | Fraction of answers with >=1 cited source |
| P95 latency (end-to-end) | 1.8s | 95th percentile of full API response time |
| P99 latency | 4.2s | 99th percentile (captures tails) |
| QPS (sustained) | 50 req/s | Sustained queries per second the system supports without degradation |
| Ingestion throughput | 200 docs/hr | Documents fully processed and indexed per hour (depends on doc size) |
| Embeddings/sec (batch) | 1000 embeddings/s | Embedding generation rate when using batched inference |
| Guardrail block rate | 2% | % of outputs blocked/modified by guardrails |
| Cost per 1k queries | $12 | Combined cost of embedding, LLM, vector-store, infra (approx.) |
| PII incidents | 0 / month | Confirmed PII leakage incidents per month (target 0)

How each metric is calculated
-----------------------------
- Precision@k: For each query in the golden set, mark whether any of the top-k retrieved chunks are labeled as relevant. P@k = (# queries with >=1 relevant in top-k) / (total queries).
- MRR: For each query, find rank r of first relevant document; reciprocal = 1/r. MRR = mean(reciprocal) across queries.
- Answer Accuracy: Use a mix of automatic checks (exact/regex matches for closed questions) and human blind labels for open answers. Accuracy = correct_count / total_sampled.
- Hallucination rate: On a sampled set, human raters mark whether claims are supported by cited sources. Hallucination rate = unsupported_count / sampled_count.
- Source attribution coverage: Count answers that include at least one source reference divided by total answers.
- Latencies (P50/P95/P99): Measured from request received to response returned via API gateway logs or Application Insights; compute percentiles over interval.
- QPS: Count of successful requests per second over a sustained window (e.g., 5 minutes average).
- Ingestion throughput & Embeddings/sec: Worker metrics aggregated from job runners or batch logs; include retries and backoff time in telemetry.
- Guardrail block rate: Structured logs from `guardrails/rails.py` showing decisions; compute blocked/modified outputs divided by total outputs.
- Cost per 1k queries: Sum cloud costs for relevant services over a period, divide by number of queries, normalize to 1k queries.

Debugging metrics by pipeline stage (what to watch and why)
--------------------------------------------------------

1) Ingestion (file fetch, parsing)
  - Key metrics: files queued, parse error rate, avg parse time, queue depth.
  - Use for debugging: high parse error → unsupported formats or broken parsers; long parse times → large files or OCR issues.
  - Improve: add parallel parsing, stream parsing, pre-flight file size/format checks.
  - Trade-offs: more parallelism increases CPU and cost; streaming reduces memory but can complicate error recovery.

2) Chunking & preprocessing
  - Key metrics: chunks generated per doc, avg chunk size, overlap %, token counts.
  - Use for debugging: poor recall often tied to too-large or too-small chunks; token overflows indicate context window issues.
  - Improve: tune chunk size/overlap, add semantic chunking via sentence boundaries.
  - Trade-offs: larger chunks improve context but reduce retrieval granularity and increase embedding size/cost.

3) Embedding generation
  - Key metrics: embeddings/sec, batch sizes, API error rate, embedding latency distribution, retry counts.
  - Use for debugging: rate-limited or high-latency embeddings limit ingestion throughput.
  - Improve: batching, parallel workers, pre-warming model instances, hybrid managed/self-hosted strategy.
  - Trade-offs: batching increases throughput but adds latency to single-document ingestion; self-hosting reduces per-call cost but increases ops complexity.

4) Vector-store writes and indexing
  - Key metrics: write latency, write throttles (e.g., Cosmos DB RU throttles), index size, partition imbalance.
  - Use for debugging: throttles or high write latency indicate need for scaling or batching; partition skew causes hotspots.
  - Improve: batched writes, partition key design, autoscaling RU/s, background compaction.
  - Trade-offs: higher provisioned throughput increases cost; aggressive batching reduces small write latency but increases batch-retry complexity.

5) Retrieval (ANN search)
  - Key metrics: recall@k, ANN recall degradation, search latency, index refresh lag.
  - Use for debugging: sudden drop in recall may indicate index corruption, wrong embedding model, or misconfiguration of ANN parameters (ef/search_k).
  - Improve: re-tune ANN index (ef/efConstruction), use exact search for critical queries, maintain hybrid index (IVF+PQ tuning).
  - Trade-offs: increasing ANN accuracy (larger index, higher ef) increases memory and search latency; exact search is slower and more expensive.

6) Prompt composition & guardrails
  - Key metrics: guardrail hit rate, prompt template failures, context token length, prompt build latency.
  - Use for debugging: high guardrail blocks could mean overly strict policies; prompt build latency contributes to end-to-end latency.
  - Improve: pre-compile prompt templates, cache commonly used prompt fragments, refine guardrail rules with thresholds.
  - Trade-offs: looser guardrails increase risk but reduce false-positive blocking; caching reduces latency but risks staleness.

7) LLM call (inference)
  - Key metrics: LLM latency (P50/P95/P99), error rate, cost per call, token usage per call.
  - Use for debugging: high tail latency may be API provider load or cold starts; high cost per call affects cost-per-query.
  - Improve: use smaller/cheaper models for non-critical requests, implement streaming responses, cache recent LLM outputs, use request shaping (reduce context tokens).
  - Trade-offs: smaller models reduce cost/latency but may lower answer quality; caching improves latency but may return stale answers.

8) Post-processing & API
  - Key metrics: response assembly time, serialization time, error rate, API gateway latency.
  - Use for debugging: slow post-processing indicates heavy formatting or verification steps.
  - Improve: move heavy processing to async jobs, return partial results with follow-ups.
  - Trade-offs: async improves responsiveness but complicates client UX and ordering semantics.

Strategies to reduce latency and their trade-offs
------------------------------------------------
- Precompute embeddings: compute embeddings at ingest time rather than on-demand.
  - Benefit: retrieval is faster; avoids embedding API calls on query path.
  - Trade-off: increases storage, reindexing cost, and freshness lag.

- Caching retrievals and LLM outputs:
  - Benefit: instant responses for repeated queries.
  - Trade-off: cache staleness and extra cache invalidation complexity; cache cost.

- Batching and asynchronous processing:
  - Benefit: higher throughput and resource efficiency for embeddings and writes.
  - Trade-off: increases latency for individual items entering the pipeline; complexity in retry and idempotency.

- Use smaller/faster models for routing or short answers:
  - Benefit: much lower latency and cost for high-frequency simple queries.
  - Trade-off: lower-quality responses for complex tasks; need routing logic.

- Dedicated inference fleet / provisioned capacity:
  - Benefit: predictable low latency and no cold-starts.
  - Trade-off: higher steady cost and more ops overhead.

- ANN tuning (ef, probes, index type):
  - Benefit: balance recall vs query latency by tuning search parameters.
  - Trade-off: increasing ef/probes increases latency and CPU/memory usage.

Types of trade-offs and how to achieve them
-----------------------------------------
- Cost vs Latency: increase provisioned capacity, use faster models, or cache results to lower latency at higher cost; or batch and use cheaper models to save cost at the expense of per-query latency.
- Accuracy vs Latency: use larger models and more context for accuracy (higher latency); or use retrieval precision tuning and smaller models for faster answers with potential accuracy loss.
- Freshness vs Consistency: synchronous reindexing yields fresh vectors but can slow ingestion; async incremental updates favor availability but may serve stale results.
- Complexity vs Control: self-hosting grants control and potential cost savings but increases operational complexity; managed services reduce ops but limit tuning.
- Security vs Convenience: stronger network isolation and encryption increase deployment complexity and latency (private endpoints, VNet), but improve compliance.

Examples: achieving one trade-off over another
--------------------------------------------
- Prefer low latency over cost: provision GPUs or dedicated instances for inference + aggressive caching + using smaller cached-models for common queries.
- Prefer cost savings over latency: use batch embeddings, lower-cost embedding models for non-critical data, and serverless autoscaling with cold starts tolerated.
- Prefer accuracy over latency: send more retrieved context, use larger LLMs, increase k for retrieval, and run post-filtering validation; accept higher P95 latency.

Operational checklist for continuous evaluation
--------------------------------------------
- Nightly regression run over `evals/golden_dataset.json` computing P@1, P@5, MRR, and hallucination sample.
- Weekly human review of sampled answers (200–500) for hallucination and usefulness.
- Alerts for embedding latency (>95th percentile threshold), vector-store throttling, and guardrail exception surge.
- Monthly cost review and query-pattern analysis to identify caching candidates.

Where to add these in the repo
------------------------------
- Add scheduled evaluation job invoking `evals/run_evals.py` and storing results in `out/` or an internal dashboard.
- Emit structured metrics from `ingestion/processor.py`, `agents/graph.py`, and `guardrails/rails.py` so Application Insights dashboards can compute these KPIs.

