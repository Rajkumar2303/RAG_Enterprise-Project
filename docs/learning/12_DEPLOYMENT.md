# Deployment — Running the RAG System in Production

## 1. What It Is

Deployment is the process of packaging, configuring, and running the RAG system in a production environment where it serves live user requests with reliability, security, and performance requirements. This covers everything from containerization and secrets management to scaling strategies and monitoring.

**Production architecture (target):**

```
                         ┌──────────────┐
                         │   Load Balancer    │
                         │   (e.g., Nginx)     │
                         └──────┬───────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
              ┌─────┴─────┐          ┌─────┴─────┐
              │ FastAPI    │          │ FastAPI    │
              │ Instance 1 │          │ Instance 2 │
              │ (Docker)   │          │ (Docker)   │
              └─────┬─────┘          └─────┬─────┘
                    │                       │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
              ┌─────┴─────┐          ┌─────┴─────┐
              │  Qdrant   │          │ Portkey    │
              │ (Managed) │          │ Gateway    │
              │ Vector DB │          │ (LLM proxy)│
              └───────────┘          └───────────┘
                    │                       │
                    │              ┌────────┴────────┐
                    │              │  OpenAI / DeepSeek│
                    │              │  (LLM provider)  │
                    │              └─────────────────┘
                    │
              ┌─────┴─────┐
              │  Postgres  │
              │  (Neon)    │
              │ Checkpointer│
              └───────────┘
```

## 2. Other Approaches

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **Monolithic container** | One container: FastAPI + everything | Simple; easy to deploy and debug | Cannot scale components independently; resource inefficient |
| **Microservices** | Separate containers for API, retrieval, ingestion | Independent scaling; clear boundaries | Complex orchestration; network overhead; harder to debug |
| **Serverless (AWS Lambda)** | Functions as backend, managed vector DB | No server management; auto-scale | Cold starts; limited timeout; harder to manage state |
| **Kubernetes** | Container orchestration with auto-scaling | Production-grade scalability; rolling updates | High operational complexity; overkill for many projects |
| **Docker Compose (this repo)** | Multi-container setup on single host | Simple; reproducible; good for staging | Single host; no auto-scaling; limited HA |

## 3. Why This Approach Was Chosen

**Why a multi-service setup with containers:**
- **Separation of concerns.** The FastAPI app, Qdrant, and Postgres each run in their own container. You can update one without affecting others.
- **Environment parity.** Docker ensures the development, staging, and production environments are identical. No more "it works on my machine."
- **Independent scaling.** In production, the API layer needs more instances than Qdrant (which handles many requests from a single connection pool). Containers make per-service scaling natural.
- **Managed services for persistence.** Qdrant Cloud and Neon Postgres are used in production, offloading storage durability, backups, and scaling to the provider.

**Why not full Kubernetes:**
- For the current project scale (single team, moderate traffic), Docker Compose + a cloud VM or container service (Azure Container Apps, AWS ECS) is sufficient. Kubernetes adds operational complexity without proportional benefit.

**Why not serverless:**
- RAG queries have unpredictable latency (retrieval + LLM call). Serverless cold starts (1-5s) would add unacceptable delay. Long-running containers are a better fit.

## 4. How Deployment Is Implemented in This Repo

### Dockerfile

The project's `Dockerfile` builds a Python 3.12 container:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

Key design choices:
- **`python:3.12-slim`** — smaller image than `python:3.12` (avoids build tools) but more compatible than `alpine` (which has C extension issues).
- **Separate `requirements-prod.txt`** — production dependencies omit test and dev packages, reducing attack surface and image size.
- **`COPY . .` after pip install** — leverages Docker layer caching: dependencies are only re-installed when `requirements-prod.txt` changes, not when source code changes.

### Docker Compose

`docker-compose.yml` defines the multi-service setup:

```yaml
services:
  api:
    build: .
    ports: ["8001:8001"]
    env_file: .env
    depends_on:
      - qdrant
  
  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]
    volumes:
      - qdrant_storage:/qdrant/storage

volumes:
  qdrant_storage:
```

This provides a fully functional local development environment. `docker compose up` starts both the API and Qdrant, with persistent vector storage.

### Environment configuration

Sensitive settings are loaded from `.env` at startup via `app/config.py`:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    
    JINA_API_KEY: str
    OPENAI_API_KEY: str
    PORTKEY_API_KEY: str
    QDRANT_URL: str
    NEON_DB_URL: str
    ...
```

Required variables raise clear errors at startup if missing. Optional variables (like `LOGFIRE_TOKEN`) have sensible defaults.

### Starting the application

```bash
# Development (direct Python)
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# Production (Docker Compose)
docker compose up --build

# Background (Docker Compose detached)
docker compose up -d
```

### Ingestion as a separate command

Ingestion is a one-shot command, not a service:

```bash
python -m app.ingestion.processor DATA/uploads
```

This runs as a batch job (CI/CD step, cron, or manual trigger). It connects to Qdrant, indexes the documents, and exits.

### Health checks

`app/health.py` provides a `/health` endpoint:

```python
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "qdrant": check_qdrant(),
        "postgres": check_postgres(),
        "redis": check_redis(),
        "guardrails": guardrails_initialized,
    }
```

Used by load balancers and monitoring to verify the service is ready.

### Connection checker

`app/services/health/connection_checker.py` runs comprehensive connection diagnostics at startup:

```python
def check_all_connections():
    checks = {
        "qdrant": check_qdrant_connection(),
        "postgres": check_postgres_connection(),
        "redis": check_redis_connection(),
        "jina_api": check_jina_api_key(),
        "portkey": check_portkey_config(),
    }
    return checks
```

If `STRICT_STARTUP=True`, the server fails to start if critical connections are unavailable.

### Observability stack

| Tool | What it monitors |
|---|---|
| **Logfire** | Distributed tracing across all steps (guardrail, planner, retriever, responder, LLM call) |
| **Prometheus** | Request counts, latency histograms, guardrail block counts |
| **LangSmith** | LangGraph step-level tracing (token usage, latency per node) |
| **Rate limiting (slowapi + Upstash Redis)** | Enforces per-IP rate limits |

## 5. How It Helps Our Project

- **Reproducible deployments.** Docker ensures that staging and production environments match development exactly.
- **Local-first development.** Run `docker compose up` on any machine and get a fully functional RAG system without manual setup.
- **Graceful startup.** The connection checker verifies all external services before accepting traffic. If Redis is down, the system falls back to in-memory rate limiting rather than crashing.
- **Observable end-to-end.** Every query is traced from guardrail → planner → retriever → reranker → LLM → response with timing and status.

## 6. What Could Be Better

- **Infrastructure as Code (IaC).** Add Terraform or Bicep templates to provision cloud infrastructure (Qdrant Cloud, Neon Postgres, container hosting, Redis) automatically.
- **CI/CD pipeline.** Automate: lint → test → build → eval → deploy. Each PR triggers the eval suite; the PR deploys only if all metrics pass.
- **Secrets management.** Currently secrets are in `.env` files. Production should use a vault (HashiCorp Vault, AWS Secrets Manager, Azure Key Vault).
- **Blue/green deployment.** Zero-downtime deployments require running two versions side by side and switching traffic when the new version passes health checks.
- **DB migration automation.** LangGraph Postgres migrations (`checkpointer.setup()`) should run as a separate init container before the API starts.
- **Auto-scaling policy.** Define CPU/memory thresholds for adding API instances, and a connection pool scaling policy for Qdrant and Postgres.

## 7. Experience Notes (Building from Scratch)

- **Docker Compose for development, not production.** Docker Compose is excellent for local dev and CI but lacks production features (rolling updates, health check-based restart, secrets management). Use it as a stepping stone to a production orchestrator.
- **Run ingestion as a CI step.** Add a CI job that runs `python -m app.ingestion.processor` and indexes the latest documents whenever the `DATA/` directory changes. This keeps the production index fresh.
- **Test the Docker build regularly.** The Dockerfile can break silently (Python package version conflicts, missing system dependencies). Include `docker build` in CI.
- **Monitor `.env` hygiene.** It's easy to accidentally commit `.env` files with real API keys. Add `.env` to `.gitignore` and use `.env.example` as a template.
- **Set resource limits.** In production, set CPU/memory limits on containers to prevent a traffic spike from exhausting server resources.
- **Health check everything.** Every external dependency (Qdrant, Postgres, Redis, Jina API, Portkey) should have a health check endpoint. Your startup should fail fast if critical dependencies are unavailable.
- **Plan for credential rotation.** API keys expire or need rotation. Design your config loading so that updating a key doesn't require a full deployment — env vars or vault reloads should trigger a graceful restart.
