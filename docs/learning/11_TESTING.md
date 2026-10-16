# Testing — Verifying Pipelines, Agents, and Model Behavior

## 1. What It Is

Testing in a RAG system spans multiple layers: unit tests for code correctness, integration tests for end-to-end pipeline behavior, and model-grounded evaluation tests for output quality. Because a RAG system involves both deterministic code (Python logic) and non-deterministic components (LLM responses), testing strategies must cover both.

**Testing layers in this project:**

| Layer | What it tests | Tools | Frequency |
|---|---|---|---|
| **Unit tests** | Individual functions (chunker, embedder, state transitions) | pytest | Every commit |
| **Integration tests** | Pipeline end-to-end (ingestion → retrieval → generation) | pytest + mocks | Every PR |
| **Guardrail tests** | Safety rules correctly block/allow inputs | pytest + NeMo | Every commit |
| **Eval tests** | RAGAS metrics, retrieval quality | evals/ suite | Before deployment |
| **Health tests** | API endpoints, connection checks | pytest + FastAPI TestClient | Every commit |

## 2. Other Approaches

| Approach | Coverage | Pros | Cons |
|---|---|---|---|
| **Unit-only** | Code paths only | Fast; easy to write | Misses model-level regressions; false confidence |
| **Integration-only** | Full pipeline | Catches real issues | Slow; flaky; hard to debug |
| **Snapshot testing** | Compare outputs to stored golden files | Catches unexpected changes | Fragile; every prompt change breaks snapshots |
| **Property-based (Hypothesis)** | Generate random inputs | Finds edge cases you didn't think of | Complex setup; overkill for simple functions |
| **LLM-as-judge eval (this repo)** | Measure answer quality with AI | Catches semantic regressions; no manual labels needed | Expensive; slow; judge LLM has biases |
| **Human review** | Manual inspection | Gold standard | Doesn't scale; subjective |

## 3. Why This Approach Was Chosen

**Why layered testing (unit + integration + eval):**
- **Fast feedback from unit tests.** The chunker, state machine, and routing logic are pure Python with no external dependencies. Unit tests for these run in milliseconds and catch most coding errors.
- **Integration tests catch wiring issues.** Does the agent graph route correctly? Do the nodes pass state as expected? Integration tests with mocked external services verify the plumbing.
- **Eval tests catch semantic regressions.** A prompt change might make answers less factual while all unit tests still pass. RAGAS evaluations detect this by measuring faithfulness and answer relevancy against a golden dataset.

**Why not snapshot testing:**
- LLM outputs are non-deterministic (even with temperature=0). Snapshot testing would produce frequent false failures when the LLM phrases the same answer differently.

**Why not only unit tests:**
- Unit tests cannot verify that the LLM uses the retrieved context correctly. A system where all unit tests pass could still produce terrible answers.

## 4. How Testing Is Implemented in This Repo

### Test directory structure

```
tests/
├── __init__.py
├── test_async_query.py       # Async API endpoint tests
├── test_auth.py              # API key authentication
├── test_checkpointer.py      # Postgres/Memory checkpointer
├── test_connection_checker.py# External service connectivity
├── test_evals.py             # Evaluation pipeline integration
├── test_health.py            # Health endpoint
├── test_metrics.py           # RAGAS metric computation
├── test_rate_limit.py        # Rate limiting
└── test_retry.py             # Retry logic (tenacity decorators)
```

### Unit test example: health endpoint

`tests/test_health.py`:

```python
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
```

FastAPI's `TestClient` makes API endpoint testing straightforward. The `client` fixture creates a fresh app instance for each test.

### Integration test example: agent graph

Testing the full agent graph requires mocking external services:

```python
def test_agent_graph_conversational_query(mock_llm, mock_qdrant):
    """A greeting should route to responder without retrieval."""
    state = AgentState(
        messages=[{"role": "user", "content": "hello"}],
        current_query="",
        documents=[],
        plan=[],
        status="",
        final_answer="",
    )
    # mock_llm returns "CONVERSATIONAL" for any input
    result = planner_node(state)
    assert result["current_query"] == "CONVERSATIONAL"
```

### Guardrail testing

Guardrail tests are critical for safety:

```python
def test_off_topic_rail_blocked():
    """"tell me a joke" should be blocked by the off-topic rail."""
    blocked, response = guard("tell me a joke")
    assert blocked is True
    assert "Enterprise IT Assistant" in response

def test_technical_query_passes():
    """A legitimate technical query should pass guardrails."""
    blocked, response = guard("how do I configure SR-IOV on Intel E810?")
    assert blocked is False
    assert response is None
```

### Evaulation as testing

The eval suite (`evals/`) serves as the final testing gate:

```bash
# Run the full eval pipeline
python -m evals.run_evals

# This:
# 1. Loads the golden dataset
# 2. Sends each question through the live RAG API
# 3. Computes RAGAS metrics (faithfulness, answer_relevancy)
# 4. Prints a summary table with scores
```

The eval suite should be run before every deployment. If scores drop significantly, the deployment is blocked.

### Mocking strategy

Because the system connects to many external services, mocking is essential for fast, reliable unit tests:

| Service | Mock strategy |
|---|---|
| **Qdrant** | `unittest.mock.patch()` — return canned search results |
| **Jina Embeddings API** | Mock `requests.post` to return fake embeddings |
| **Jina Reranker API** | Same mock approach as embeddings |
| **LLM (Portkey/DeepSeek)** | Mock `client.chat.completions.create` to return predefined responses |
| **Postgres (Neon)** | Use SQLite in-memory or skip checkpointer tests |
| **Redis (Upstash)** | Skip or use a local Redis mock |

## 5. How It Helps Our Project

- **Catches code regressions.** Changes to the chunker, agent graph, or routing logic are caught by unit tests before reaching production.
- **Verifies safety.** Guardrail tests prove that off-topic and jailbreak inputs are blocked, and that legitimate queries pass through.
- **Quantifies quality changes.** The eval suite measures whether a change improved or degraded answer quality. This prevents "improvements" that actually make things worse.
- **Documents expected behavior.** Test names like `test_off_topic_rail_blocked` serve as living documentation of system behavior.

## 6. What Could Be Better

- **CI integration.** Run `pytest` on every PR and block merges on test failures. Run the eval suite nightly and alert on regressions.
- **Test coverage reporting.** Use `pytest-cov` to measure line coverage and set a minimum threshold (e.g., 80%).
- **Adversarial tests.** Add tests for edge cases: empty input, extremely long input, Unicode injection, SQL injection in metadata fields, concurrent requests.
- **Load testing.** The `scripts/locustfile.py` (if present) should be run regularly to verify the system handles expected traffic without degradation.
- **Deterministic LLM testing.** For unit tests, replace the LLM with a deterministic stub that returns predefined responses. This makes tests faster and flake-free.
- **Regression dataset.** When a bug is found in production, add the failing query to the golden dataset so the same bug cannot recur.

## 7. Experience Notes (Building from Scratch)

- **Write guardrail tests first.** Safety-critical checks should be tested before the rest of the system. If guardrails fail, nothing else matters.
- **Mock external services aggressively.** Tests that hit real APIs are slow, flaky, and expensive. Mock everything except in dedicated integration test runs.
- **Use fixtures wisely.** FastAPI's `TestClient` fixture creates an isolated app instance. Use it for most API tests.
- **Don't test the LLM's intelligence.** Don't write tests like `assert "Kubernetes" in response` — the LLM might answer correctly without mentioning "Kubernetes" explicitly. Instead, test structure (response format, status codes, field presence) and use the eval suite for quality.
- **Run tests in CI on every push.** A PR that breaks tests should not be merged. Configure GitHub Actions (or equivalent) to run `pytest -q` automatically.
- **Keep tests fast.** If the full test suite takes more than 30 seconds, developers will skip running it. Separate fast unit tests from slow integration/eval tests, and run the latter only before deployment.
- **Test the checkpointer.** If the Postgres checkpointer fails, agent state is lost on restart. Write a test that verifies checkpointing and restoration produce the same state.
- **Golden dataset maintenance.** Review and update the golden dataset periodically. Questions that were relevant six months ago may no longer apply, and new use cases need new test questions.
