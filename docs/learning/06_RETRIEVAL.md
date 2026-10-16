# Retrieval — Finding Relevant Evidence

## 1. What It Is

Retrieval is the process of, given a user query, selecting the most relevant passages from a large corpus. In a RAG system, this is the step that determines what the LLM will see — if retrieval misses the correct passage, even the best LLM cannot produce the right answer.

**Key properties of a good retrieval system:**
- **Recall** — does it find the relevant passages at all? (top-k retrieval)
- **Precision** — among the retrieved passages, how many are actually relevant? (top-k quality)
- **Latency** — how fast can it return results? (critical for interactive applications)
- **Scalability** — can it handle millions of documents?

**The retrieval pipeline in this repo:**

```
User Query ("how do I configure SRIOV on Intel E810")
    │
    ▼
┌───────────────────────┐
│ 1. Embed Query         │  ← embed_query(query) via Jina API
│    (query → vector)    │      or local fallback model
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 2. Vector Search       │  ← search_enterprise_knowledge(query)
│    Qdrant query_points │      ANN search, limit=15 candidates
│    (ANN, top-15)       │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 3. Semantic Reranking  │  ← rerank_documents(query, docs, top_n=5)
│    Jina Reranker API   │      Cross-encoder scoring
│    (top-15 → top-5)    │      Better precision than vector alone
└───────────┬───────────┘
            ▼
        Top-5 relevant passages (with scores)
```

## 2. Other Approaches

| Approach | How it works | Strengths | Weaknesses |
|---|---|---|---|
| **BM25 (keyword)** | Sparse TF-IDF with term frequency saturation | Very fast; no GPU/model needed; perfect for exact term matching | Zero semantic understanding; "car" ≠ "automobile"; fails on paraphrases |
| **Vector (dense)** | Semantic embedding + ANN search | Understands meaning; handles synonyms; good recall | Requires embedding model; slower at index time; may miss exact keywords |
| **Hybrid (BM25 + vector)** | Combine scores from both | Best of both worlds; high recall + precision | More complex to tune weight; two query paths; higher latency |
| **Learned sparse (SPLADE)** | Learned sparse embeddings | Combines interpretability of sparse with semantic matching | Complex to train/deploy; fewer managed services |
| **ColBERT (late interaction)** | Token-level matching via MaxSim | Very high precision for passage retrieval | Higher storage cost; slower search |
| **Graph-based (KNN via HNSW/IVF)** | Approximate nearest neighbor | Fast search at billion-scale | Index building is slow; recall vs. speed tradeoff |

## 3. Why This Approach Was Chosen

**Why pure vector search + reranker:**
- **Vector search for recall.** The enterprise documents cover specialized domains (Kubernetes, Intel hardware, networking). Users may ask "how do I enable SR-IOV" when the document says "Single Root I/O Virtualization configuration." Vector embeddings bridge this vocabulary gap naturally.
- **Reranker for precision.** ANN vector search returns 15 candidates; the reranker (cross-encoder) re-scores them with much higher accuracy, picking the top-5 most relevant. This two-stage approach gives the best of both: high recall from fast ANN, high precision from the reranker.
- **Simplicity.** A two-stage dense pipeline (embed → ANN → rerank) is simpler to maintain than a hybrid system that requires two separate indexes and score fusion logic.

**Why not hybrid:**
- Hybrid search (BM25 + vector) would improve recall for queries that contain specific keywords absent from semantically similar vectors. However, it adds complexity: you need both a keyword index and a vector index, plus a fusion strategy (e.g., reciprocal rank fusion). The current pipeline achieves acceptable quality without this complexity.

**Why not pure keyword:**
- Users express the same technical concept in many ways. BM25 would fail on "how to set up virtual functions on Intel NIC" vs. "SRIOV VF configuration." The project's domain vocabulary is too varied for keyword-only retrieval.

## 4. How Retrieval Is Implemented in This Repo

### Vector search (Qdrant)

`app/services/retrieval/qdrant_service.py`:

```python
def search_enterprise_knowledge(query: str, limit: int = 8):
    query_vector = embed_query(query)
    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    results = []
    for res in response.points:
        results.append({
            "content": res.payload.get("text", ""),
            "source": res.payload.get("source", "Unknown"),
            "score": res.score,
        })
    return results
```

Key design decisions:
- **Modern `query_points` API** (not legacy `search`) — this is the current Qdrant standard.
- **Returns 15 candidates** (hardcoded limit in the retriever node) then reranker reduces to 5.
- **Retry with exponential backoff** — the `_search_enterprise_knowledge` inner function is decorated with `@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))`.
- **Graceful degradation** — if Qdrant is unreachable after retries, returns an empty list rather than crashing.

### Embedding the query

`app/services/retrieval/embedding.py` `embed_query()` function:

```python
def embed_query(query: str) -> list[float]:
    return _embed([query], task="retrieval.query")[0]
```

The `task` parameter tells the Jina API to optimise the embedding for query-side vs. document-side representation (asymmetric embedding). This improves retrieval accuracy compared to using the same embedding function for both.

### Reranker (Jina Reranker API)

`app/services/retrieval/ranking_service.py`:

```python
def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    ranker = _JinaReranker()
    reranked_docs = ranker.rerank(query, documents, top_n)
    return reranked_docs
```

The Jina reranker is a cross-encoder model (`jina-reranker-v3`) that scores each (query, document) pair independently. This gives much higher relevance accuracy than the dot-product similarity used in ANN search.

### Integration with agents

The retriever is a LangGraph node (`app/agents/nodes/retriever.py`):

```python
def retrieve_node(state: AgentState):
    raw_results = search_enterprise_knowledge(query, limit=15)
    doc_contents = [doc["content"] for doc in raw_results]
    reranked_contents = rerank_documents(query, doc_contents, top_n=5)
    formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]
    return {"documents": formatted_docs, ...}
```

The planner node decides *whether* to call retrieval (technical queries) or skip it (conversational queries). If the planner outputs "CONVERSATIONAL", the graph routes directly to the responder with no retrieval.

## 5. How It Helps Our Project

- **Semantic understanding.** The system finds relevant passages even when the user's phrasing differs from the document's wording. A query about "NIC virtual functions" matches documentation about "SR-IOV VF configuration."
- **Two-stage quality.** The initial ANN search casts a wide net (15 candidates), then the reranker filters aggressively (top-5). This catches edge cases that pure top-5 vector search would miss.
- **Controlled latency.** ANN search in Qdrant is typically <100ms, and the reranker processes 15 documents in ~1-2 seconds. Total retrieval latency is ~2s, acceptable for an interactive system.
- **Graceful degradation.** If the reranker API fails (network timeout, rate limit), the code falls back to the original Qdrant ordering, ensuring the user still gets an answer.
- **Observability.** Each step (embedding, search, reranking) is instrumented with logfire spans, so you can identify bottlenecks.

## 6. What Could Be Better

- **Hybrid search.** As mentioned, adding BM25 would improve recall for exact- keyword queries. Qdrant supports hybrid search natively via `query_points` with a `prefetch` parameter.
- **Caching.** Frequently asked queries produce the same retrieval results. Caching the top-5 results (keyed by query embedding hash) would reduce latency and API costs.
- **Query rewriting.** The planner could rewrite the user's query before retrieval (e.g., expand acronyms, fix typos). This is especially useful for technical queries with abbreviations.
- **Multi-vector retrieval.** For complex queries, decompose into sub-queries, retrieve for each, then merge results. This improves recall for multi-faceted questions.
- **Metadata filtering.** The Qdrant payload includes `source_type` (noisy/true/general). Adding pre-filtering by source type would let users restrict search to trusted documents only.

## 7. Experience Notes (Building from Scratch)

- **Evaluate recall@k first.** Before adding a reranker, measure how often the correct passage appears in the top-5, top-10, and top-20 from pure vector search. If recall@20 is below 80%, fix chunking or embedding before investing in reranking.
- **Golden dataset is essential.** Create 20-50 Q&A pairs where each question maps to the exact passage containing the answer. Use this dataset to evaluate every retrieval change.
- **Test with synonyms.** Ensure your test queries include paraphrases: "virtual functions on Intel NIC" and "SRIOV VF configuration" should both retrieve the same passage. If not, your embedding model may not capture the domain vocabulary well.
- **Monitor reranker latency.** Cross-encoders are slow by nature. If latency exceeds 3s, consider reducing the candidate pool (10 instead of 15) or switching to a smaller reranker model.
- **Handle empty results gracefully.** When the corpus is empty (first run) or the query is out-of-domain, the retriever returns empty. The responder should detect this and say "I don't have information about that in my knowledge base" rather than hallucinating.
- **Log retrieval quality.** Log the query, top-5 passages, and their scores. This lets you retrospectively analyze why a bad answer was generated — was it a retrieval miss or a generation error?
- **Plan for refresh lag.** If new documents are ingested but the vector index hasn't been rebuilt, queries won't return them. Either use incremental indexing or clearly communicate the refresh schedule to users.
