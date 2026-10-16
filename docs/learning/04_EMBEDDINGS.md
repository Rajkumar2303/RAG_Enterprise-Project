# Embeddings — Representing Text as Vectors

## 1. What It Is

Embeddings are dense numerical vectors (lists of floating-point numbers) that capture the semantic meaning of a piece of text. In a RAG system, embeddings serve as the universal language that connects user queries to document passages — both are converted to vectors, and similarity is measured by how close those vectors are in the embedding space.

**Key properties:**
- **Fixed dimensionality** — typically 384, 768, 1024, or 1536 floats regardless of input length.
- **Semantic proximity** — similar meanings produce similar vectors even if they use different words ("SR-IOV configuration" ≈ "setting up virtual functions on NIC").
- **Directional** — cosine similarity or dot product measures alignment; magnitude indicates confidence.
- **Contextual** — modern embedding models are transformers that consider surrounding context, not just individual words.

**How embeddings are used in this project:**
1. **Offline (ingestion):** Each document chunk → embedding vector → stored in Qdrant.
2. **Online (query):** User query → embedding vector → ANN search against stored vectors.
3. **Asymmetric embedding:** Queries and documents use different "task" embeddings even from the same model, because a query ("how do I configure X?") looks different from a document passage ("To configure X, follow these steps..."). Jina supports this via the `task` parameter.

## 2. Other Approaches

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **TF-IDF (bag of words)** | Term frequency × inverse document frequency | Interpretable; fast; no model needed | No semantics; zero for out-of-vocabulary terms; sparse |
| **BM25 (OKapi)** | TF-IDF with term saturation and length normalization | State-of-the-art sparse; best for exact-match | Still no semantic understanding |
| **Word2Vec / GloVe** | Static word embeddings averaged | Lightweight; well-understood | No context; "bank" (river) vs "bank" (finance) are the same vector |
| **Sentence-BERT / SBERT** | Bi-encoder transformer | Good semantic understanding; fast to compute | Needs GPU for large batches; moderate quality ceiling |
| **Instructor / Jina-v3** | Instruction-tuned embeddings | Task-aware (query vs. doc); state-of-the-art | API cost (if using hosted); larger model size |
| **OpenAI embeddings (text-embedding-3-*)** | API-based, 1536/3072 dimensions | Very high quality; simple API | Vendor lock-in; cost per token; rate limits |
| **Cohere embeddings** | API-based, supports multilingual | Good for non-English content | Same vendor lock-in concerns |
| **Learned sparse (SPLADE)** | Learned sparse vectors | Combines interpretability with semantics | Harder to deploy; fewer vector DBs support it natively |

## 3. Why This Approach Was Chosen

**Why Jina Embeddings v3 (API-first, with local fallback):**
- **Jina v3 offers task-aware encoding.** The same model produces different embeddings for queries vs. documents (`retrieval.query` vs. `retrieval.passage`). This is proven to improve retrieval accuracy over naive single-vector approaches.
- **1024-dimension vectors** — a good tradeoff between precision and storage/search cost. 1536 dimensions (OpenAI) would be more precise but slower and more expensive; 384 dimensions (miniLM) would be faster but less accurate.
- **Local fallback (mxbai-embed-large-v1).** If the Jina API is unreachable or the API key is missing, the system transparently falls back to a local SentenceTransformer model. This is critical for development environments without internet access.
- **Batch processing.** The embedder sends batches of 64 texts to the API, balancing throughput and API rate limits.

**Why not OpenAI/Cohere:**
- OpenAI embeddings would have tied the project to OpenAI's ecosystem and pricing. Jina offers a comparable API with similar quality.
- Cohere was not chosen because Jina's task-aware encoding was more straightforward to integrate.

**Why not a pure local model:**
- A local SentenceTransformer model (even mxbai-embed-large-v1 at 1024-dim) is significantly less accurate than Jina v3 for domain-specific technical text. The project values retrieval quality over full offline independence.

## 4. How Embeddings Are Implemented in This Repo

### Architecture overview

`app/services/retrieval/embedding.py`:

```
_init()
    ├── _probe_jina_api()  → if OK, _model_type = "jina"
    └── (else)             → _load_fallback(), _model_type = "fallback"
    
embed_texts(texts, task)  → _embed(texts, task)
    ├── if jina:   _embed_jina(texts, task)
    │                 → batches of 64
    │                 → _embed_jina_batch() with retry
    └── if fallback: _embed_fallback(texts)
                      → _active_model.encode()
```

### Jina API embedding

```python
def _embed_jina_batch(texts, task):
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.JINA_API_KEY}"},
        json={
            "model": "jina-embeddings-v3",
            "task": task,         # "retrieval.query" or "retrieval.passage"
            "normalized": True,
            "input": texts,
        },
        timeout=60,
    )
    # Sort by index to ensure order preservation
    results_sorted = sorted(results, key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in results_sorted]
```

Key details:
- **`normalized: True`** — returns unit vectors so cosine similarity = dot product (faster search).
- **`task` parameter** — crucial for asymmetric retrieval. Queries use `retrieval.query`, documents use `retrieval.passage`.
- **Exponential backoff retry** — 3 attempts with 1-5 second wait.
- **Batch size 64** — optimizes throughput without hitting payload size limits.

### Local fallback embedding

```python
def _load_fallback():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
```

The fallback model produces the same 1024-dim vectors but without task-aware encoding. It is used when the Jina API is unreachable.

### Runtime fallback mechanism

```python
def _embed(texts, task):
    _init()
    if _model_type == "jina":
        try:
            return _embed_jina(texts, task)
        except Exception as e:
            logfire.error(f"Jina API failed: {e}. Switching to local model.")
            _ensure_fallback()
    return _embed_fallback(texts)
```

This ensures zero-downtime embedding: if the API goes down mid-ingestion, the remaining chunks use the local model without manual intervention.

### Query-time embedding

```python
def embed_query(query: str) -> list[float]:
    return _embed([query], task="retrieval.query")[0]
```

Called from `qdrant_service.py` before every vector search.

## 5. How It Helps Our Project

- **Semantic retrieval.** The system understands that "SR-IOV virtual functions" and "PCIe passthrough VF configuration" are related concepts, even though they share few surface words.
- **Task-aware accuracy.** Using different embeddings for queries vs. passages improves retrieval precision compared to symmetric models.
- **Resilience.** If Jina API has an outage, ingestion and querying continue with the local fallback. Users experience a slight quality degradation but no downtime.
- **Consistent dimensionality.** Both models produce 1024-dim vectors, so the index doesn't need to be rebuilt when switching between them.
- **Cost-effective.** Batching and caching reduce API calls. The fallback model costs nothing to run.

## 6. What Could Be Better

- **Domain-specific fine-tuning.** The current embedding models are general-purpose. Fine-tuning on enterprise technical documentation (Kubernetes, networking) could improve retrieval accuracy for domain-specific queries.
- **Caching.** Frequently embedded texts (common queries, unchanged documents) could be cached to avoid redundant API calls.
- **Multi-model embeddings.** Some queries benefit from different embedding models (e.g., code snippets vs. prose). A router that selects the best embedding model per query type could improve overall quality.
- **Dimensionality reduction.** 1024 dimensions works well, but for very large corpora, reducing to 256-512 dimensions with a technique like PCA could speed up search with minimal quality loss.
- **Quantization.** Converting float32 vectors to int8 would reduce storage by 4x and speed up search, with minimal accuracy impact.

## 7. Experience Notes (Building from Scratch)

- **Always normalize.** Use normalized embeddings so cosine similarity equals dot product. This simplifies search logic and ensures consistent scoring ranges.
- **Keep query and passage embeddings separate.** Never use the same embedding for both indexing and querying if your model supports task-specific embeddings. It makes a measurable difference in retrieval quality.
- **Test embedding quality early.** Before building the full RAG pipeline, write a small test: take 10 documents, write 5 queries for each, embed both, and verify that the top-3 results by cosine similarity are actually relevant.
- **Monitor embedding drift.** If you change embedding models or versions, old and new vectors will have different distributions. You must re-index all documents to maintain consistent search quality.
- **Batch wisely.** Most embedding APIs have throughput limits. A batch size of 64 is a good starting point; adjust based on API response times and error rates.
- **Plan for cold start.** When the system starts with an empty corpus, the first few queries will return nothing. Either seed the corpus with starter documents or surface a "No documents indexed yet" message.
- **Log embedding failures.** If an embedding call fails, log the input text length and model type. This helps diagnose API rate limiting or model-specific input length limits (Jina v3 supports up to 8192 tokens per input).
