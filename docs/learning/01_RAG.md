# RAG — Retrieval-Augmented Generation

## 1. What It Is

Retrieval-Augmented Generation (RAG) is an architecture that enhances a large language model's output by first retrieving relevant documents or passages from an external knowledge base and then feeding that retrieved context into the LLM's prompt. The model generates its answer conditioned on both the user query and the retrieved evidence, rather than relying solely on what it memorised during training.

**Core idea:** separate parametric memory (the weights of the LLM) from non-parametric memory (an external corpus that can be updated independently). When a question arrives, the system:
1. Converts the query into a vector (embedding).
2. Searches a vector database for the most semantically similar passages.
3. Assembles those passages into a prompt alongside the user question.
4. Calls the LLM, which reads the context and produces a grounded answer.

**Key properties:**
- **Groundedness** — answers are backed by specific source passages; the LLM is not guessing.
- **Freshness** — update the corpus without retraining the model.
- **Explainability** — every answer can be traced back to the documents that influenced it.
- **Modularity** — swap retrieval backends, embedding models, or LLM providers independently.

## 2. Other Approaches

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **Pure generation (zero-shot)** | LLM answers from its weights alone | Simplest setup; no infrastructure needed | Hallucinations; stale knowledge; no provenance |
| **Fine-tuning** | Train model weights on domain data | Deeper integration; can learn style/format | Expensive; needs curated data; still hallucinates on rare facts; must re-tune when data changes |
| **Prompt engineering (few-shot)** | Put examples in the prompt | Cheap; fast to iterate | Context window limits number of examples; no external evidence |
| **Agentic tool use** | LLM decides when to call a search tool | Flexible; can handle many tool types | Complex orchestration; latency from sequential calls |
| **Memorisation KB (non-neural)** | Traditional DB + query | Deterministic; full control over results | No semantic matching; brittle for natural language questions |

## 3. Why RAG Was Chosen (and Why Not Others)

**Why RAG:**
- **Up-to-date without retraining.** Enterprise documents change frequently (new specs, updated policies). RAG lets us point the system at fresh data and immediately get correct answers without a fine-tuning cycle.
- **Explainability and audit.** In an enterprise setting, stakeholders need to know *why* the system answered a certain way. RAG naturally surfaces source documents. Pure generation cannot do this.
- **Cost.** Fine-tuning large models (GPT-4 class) is very expensive and must be repeated. RAG uses a single general-purpose LLM with domain context, so the same model serves many domains.
- **Rapid iteration.** Changing chunking strategy, embedding model, or prompt template takes minutes, not days of training.

**Why not others:**
- **Not pure generation** — hallucination risk is too high for technical documentation (Kubernetes, Intel hardware, networking). A wrong config value could cause production outages.
- **Not fine-tuning** — the data in this project changes; fine-tuning would need to be repeated on every update, which is slow and costly. Also, fine-tuned models still need RAG for truly up-to-date retrieval.
- **Not pure tool-use agents** — simpler RAG is sufficient for factual Q&A; full agentic loops add latency without proportional benefit for straightforward retrieval.

## 4. How RAG Is Implemented in This Repo

The full RAG pipeline spans several modules:

```
User Query
    │
    ▼
┌─────────────────────┐
│   Guardrails (gate)  │  ← app/guardrails/rails.py — quick safety check
│   (optional reject)  │
└─────────┬───────────┘
          │ (pass)
          ▼
┌─────────────────────┐
│   Planner (agent)    │  ← app/agents/nodes/planner.py
│   Intent detection   │      Decides: conversational or needs retrieval?
└─────────┬───────────┘
          │ (technical)
          ▼
┌─────────────────────┐
│   Retriever node     │  ← app/agents/nodes/retriever.py
│   Qdrant vector DB   │      Calls search_enterprise_knowledge()
│   + reranker (Jina)  │      Then rerank_documents()
└─────────┬───────────┘
          │ (top-5 passages)
          ▼
┌─────────────────────┐
│   Responder (gen)    │  ← app/agents/nodes/responder.py
│   Assembles prompt   │      Concatenates context + history → LLM
│   Calls LLM gateway  │      app/gateway/client.py (Portkey/DeepSeek)
└─────────┬───────────┘
          │ (raw answer)
          ▼
┌─────────────────────┐
│   (Guardrails post)  │  ← Optional second pass
│   Final answer       │
└─────────────────────┘
```

**Key files:**
- `app/agents/graph.py` — LangGraph state graph wiring the nodes.
- `app/agents/nodes/planner.py` — intent detection; outputs `CONVERSATIONAL` or a search query.
- `app/agents/nodes/retriever.py` — calls Qdrant + Jina reranker to get top-5 passages.
- `app/agents/nodes/responder.py` — prompt construction and LLM call with retry logic.
- `app/gateway/client.py` — LLM client abstraction (Portkey with fallback to DeepSeek).
- `app/guardrails/rails.py` — NeMo Guardrails for input/output safety.
- `app/ingestion/processor.py` — offline pipeline for building the vector index.

**State management:** `app/agents/state.py` defines an `AgentState` TypedDict with fields for messages, documents, plan, status, and final answer. Messages are accumulated using `Annotated[list, operator.add]` so LangGraph appends rather than replaces.

**Retry and resilience:** The responder uses `tenacity` with exponential backoff (3 attempts, 1s-5s wait). The retriever also retries on Qdrant failures and gracefully returns empty results on persistent errors.

## 5. How It Helps Our Project

- **Grounded enterprise answers.** The system answers questions about Kubernetes, Intel hardware, and networking using actual documentation — not the LLM's best guess.
- **Multi-turn conversations.** The planner distinguishes between chit-chat ("hi") and technical queries ("how do I configure SRIOV on an Intel E810?"), routing appropriately.
- **Audit trail.** Every answer includes the source passages that were retrieved; users can verify claims.
- **Cache acceleration.** Portkey gateway caches LLM responses for identical queries, reducing latency and cost.
- **Graceful degradation.** If the vector DB is down, retrieval returns empty and the LLM can still answer conversationally; if the LLM fails, the user gets a clear error instead of a silent failure.

## 6. What Could Be Better

- **Hybrid search.** Currently only vector search is used. Adding BM25 (keyword) alongside vector similarity would improve recall for precise queries like config flag names ("--kubelet-args").
- **Cross-encoder reranker.** The Jina reranker is a good start; a cross-encoder (e.g., Cohere rerank or BGE-reranker) could further improve top-k precision.
- **Query expansion.** For complex multi-part questions, expand the query into sub-queries before retrieval (e.g., "how do I set up monitoring and alerting" → two retrievals).
- **Continuous ingestion.** The current pipeline runs on-demand. A watch mode that picks up new files and incrementally indexes them would keep the knowledge base fresh.
- **Citation extraction.** After generation, extract which passages the LLN actually used and surface inline citations (e.g., [1], [2]) linked to the source documents.

## 7. Experience Notes (Building from Scratch)

- **Start minimal.** Begin with a single script: load a file → chunk → embed → FAISS index → naive similarity search → print prompt + LLM output. Get one end-to-end cycle working before adding agents, guardrails, or rerankers.
- **Evaluate early.** Collect a small golden dataset (10-20 Q&A pairs) before tuning chunking or prompt templates. Measure recall@k and answer correctness after every change so you know what works.
- **Chunk size matters a lot.** Test 256, 512, 1024, and 1500 tokens with 10% overlap. Too small → context is fragmented. Too large → each chunk contains noise. The sweet spot depends on your documents and the LLM's context window.
- **Prompt template design.** Use a consistent template that clearly separates retrieved context from the user question. Example:
  ```
  You are a {role}. Answer the question using ONLY the context below.
  
  CONTEXT:
  {retrieved_chunks}
  
  QUESTION: {user_query}
  ```
  Always prepend "I don't know" instruction to reduce hallucinations.
- **Monitor caching.** Portkey caching can mask changes: if you update an embedding model or prompt but the cache hasn't expired, you'll still see old behavior. Add a cache-busting header during development.
- **Watch token budgets.** Retrieved context can easily exceed the LLM's context window. Implement dynamic truncation that keeps the most relevant chunks and drops low-scoring ones.
- **Log everything.** Logfire spans track every step (guardrail → planner → retriever → reranker → LLM). Without observability, diagnosing a bad answer is guesswork.
