# Vector Store — Storing and Querying Embeddings

## 1. What It Is

A vector store (or vector database) is a specialized system designed to persist embedding vectors alongside their associated metadata, and to efficiently retrieve vectors that are most similar to a given query vector using approximate nearest neighbor (ANN) or exact search algorithms.

**Core responsibilities:**
- **Storage.** Persist vectors + metadata with durability and replication guarantees.
- **Indexing.** Build an efficient data structure (HNSW, IVF, PQ) that enables fast approximate search.
- **Querying.** Given a query vector, return the top-k nearest vectors ranked by distance metric (cosine, dot product, Euclidean).
- **Filtering.** Optionally filter results by metadata before or after search (e.g., `source_type == "true"`).
- **CRUD operations.** Insert, update, delete individual vectors without rebuilding the entire index.

**Why a dedicated vector store instead of a regular database?**
- Traditional databases (PostgreSQL, MySQL) can store vectors but cannot efficiently search them at scale because B-trees don't work in high-dimensional spaces.
- Specialised ANN algorithms (HNSW, IVF-PQ) achieve sub-second search over millions of vectors by trading a small amount of accuracy for massive speed gains.

## 2. Other Approaches

| Solution | Type | Pros | Cons |
|---|---|---|---|
| **FAISS (local)** | Library (in-process) | Free; fast; no network; great for dev | No persistence by default; single-machine; manual sharding |
| **Qdrant** | Managed / Self-hosted | Purpose-built; rich filtering; REST+gRPC; built-in HNSW | Operation cost; network latency |
| **Pinecone** | Managed SaaS | Zero-ops; auto-scaling; high availability | Vendor lock-in; expensive at scale; no self-host option |
| **Weaviate** | Managed / Self-hosted | GraphQL API; hybrid search built-in | More complex to operate; moderate performance |
| **Milvus** | Managed / Self-hosted | Cloud-native; GPU-accelerated; massive scale | Steep learning curve; heavy resource requirements |
| **pgvector (PostgreSQL)** | Extension | No extra infra; ACID transactions | Slower ANN; limited scalability for >1M vectors |
| **Redis Stack** | Module | Extremely fast in-memory | RAM costs; limited filtering; no persistence guarantee |

## 3. Why Qdrant Was Chosen

**Why Qdrant (over alternatives):**

- **Purpose-built for vectors.** Unlike pgvector, Qdrant is designed from the ground up for vector search. Its HNSW implementation is highly optimized and mature.
- **Rich filtering.** Qdrant supports payload filtering (pre-filtering by metadata before vector search), which we need for source-type segmentation ("noisy" vs "true" data).
- **Managed option.** Qdrant Cloud offers a managed SaaS tier with minimal operational overhead, similar to Pinecone but with self-host option as backup.
- **Native REST + gRPC.** The Python client is well-maintained and uses gRPC for high-throughput ingestion.
- **Cost-effective.** For our scale (tens of thousands of chunks), the free Qdrant Cloud tier or a self-hosted Docker instance is sufficient. Pinecone would be more expensive at equivalent scale.
- **Modern API.** Qdrant's `query_points` API (used in this project) is cleaner and more flexible than the legacy `search` API.

**Why not others:**
- **FAISS** — great for development but lacks persistence and filtering. Used as a fallback concept (via the embedding module's local model) but not as the primary store.
- **Pinecone** — vendor lock-in (no self-host), higher cost, and no pre-filtering on metadata at the time of evaluation.
- **pgvector** — excellent for PostgreSQL users, but our team preferred a purpose-built vector database with better ANN performance.
- **Milvus** — too heavy for our current scale; requires Kubernetes and significant operational investment.

## 4. How Qdrant Is Implemented in This Repo

### Client initialization

`app/ingestion/processor.py` and `app/services/retrieval/qdrant_service.py` share the same client init:

```python
from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)
```

Settings come from `app/config.py`:
```python
QDRANT_URL: str = Field(validation_alias=AliasChoices("QDRANT_URL", "QDRANT_CLUSTER_ENDPOINT"))
QDRANT_API_KEY: str | None = None
QDRANT_COLLECTION: str = "enterprise_rag"
```

Note: `QDRANT_API_KEY` accepts an empty string and converts it to `None`, so local Qdrant (no auth) works without modification.

### Collection schema

The collection `enterprise_rag` stores:
- **Vector:** 1024-dimensional float32 (from Jina v3 or mxbai fallback)
- **Payload (metadata):**
  - `text` (str) — the original chunk text
  - `source` (str) — filename
  - `source_type` (str) — "true", "noisy", "general"

### Indexing (ingestion)

In `app/ingestion/processor.py`, after chunking and embedding:

```python
points = [
    models.PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "text": chunk,
            "source": filename,
            "source_type": source_type,
        },
    )
    for chunk, vector in zip(chunks, embeddings)
]
qdrant_client.upsert(
    collection_name=settings.QDRANT_COLLECTION,
    points=points,
)
```

Key details:
- **UUID as point ID** — random UUIDs avoid collisions when re-indexing.
- **Batch upsert** — the SDK handles batching internally (up to 64 points per batch by default).
- **Payload for every point** — includes the original text so the retriever doesn't need a separate metadata store.

### Querying

In `app/services/retrieval/qdrant_service.py`:

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

Key details:
- **`query_points` (not `search`)** — the modern Qdrant API.
- **`with_payload=True`** — returns both vector and metadata in one call.
- **Score included** — downstream components can use the score for filtering or weighting.
- **Retry wrapper** — `@retry` with exponential backoff for transient failures.

## 5. How It Helps Our Project

- **Persistent knowledge base.** Once indexed, documents remain searchable across restarts. No need to re-process on every deployment.
- **Fast ANN search.** Qdrant's HNSW index returns top-15 results in <100ms for our corpus size.
- **Metadata filtering.** The `source_type` field in the payload enables future filtering (e.g., "only search trusted documents").
- **CRUD flexibility.** To update an index entry, you upsert a new point with the same ID. To remove outdated content, you delete by ID or by filter.
- **Managed or self-hosted.** The same Python client works against both Qdrant Cloud and a local Docker instance, so development and production use the same code.

## 6. What Could Be Better

- **Index optimization.** Qdrant uses default HNSW settings. Optimizing `m` (graph degree) and `ef_construct` would improve recall/speed tradeoff for our specific dataset size and query distribution.
- **Sharding.** If the corpus grows beyond a few million vectors, we'd need to shard across multiple Qdrant nodes. Qdrant supports this natively but requires cluster mode.
- **WAL/replication.** For high-availability production, configure Qdrant with write-ahead-log replication. Currently, a single node restart causes brief read unavailability.
- **Payload indexing.** Qdrant can index payload fields for faster filtering. If we add heavy metadata filtering (date ranges, categories), we should create indexes on those fields.
- **Backup strategy.** Periodic snapshots of the Qdrant collection should be taken to prevent data loss. Qdrant supports full collection snapshots via API.

## 7. Experience Notes (Building from Scratch)

- **Start with FAISS for prototyping.** Before setting up Qdrant (Cloud or Docker), use a local FAISS index in a Jupyter notebook to iterate on chunking, embedding, and retrieval quality. Only deploy Qdrant once the pipeline is stable.
- **Use the same dimension throughout.** Changing embedding models mid-project requires re-indexing. Pick your dimension (1024 is a good default) and stick with it.
- **Set up Qdrant via Docker first.** Before buying a managed plan, run Qdrant locally with Docker:
  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```
  This gives you a fully functional vector store to develop against, costing nothing.
- **Test empty collection behavior.** When the collection doesn't exist yet, Qdrant's `query_points` returns empty results (not an error). Make sure your code handles this gracefully.
- **Monitor disk usage.** Qdrant stores vectors in-memory for fast search, but also persists to disk. For 10K vectors at 1024 dimensions, expect ~80MB RAM + disk per replica.
- **Payload size matters.** Each point's payload (text + metadata) is stored alongside the vector. Large payloads increase storage and slow down searches. Keep payloads concise — store only what the retriever needs.
- **Delete before re-indexing.** The `run_universal_ingestion` function supports `wipe=True` which calls `qdrant_client.delete_collection()` + `recreate_collection()`. Always wipe before re-indexing with a different embedding model to avoid mixing incompatible vectors.
