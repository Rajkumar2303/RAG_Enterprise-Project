# Ingestion — Reading and Preparing Data for RAG

## 1. What It Is

Ingestion is the offline pipeline that takes raw documents (PDFs, HTML pages, Word documents, plain text files) and transforms them into a searchable, indexed knowledge base. It is the foundation upon which the entire RAG system rests — if ingestion is broken, retrieval and generation will both produce poor results.

**The ingestion pipeline consists of these stages:**

```
Raw Files (DATA/uploads, DATA/noisy_data, DATA/true_data)
    │
    ▼
┌──────────────┐
│ File Loader  │ ← Detects file type and calls the appropriate parser
│ (extractors) │     PDF → pdfplumber/pypdf
└──────┬───────┘     HTML → beautifulsoup/lxml
       │              TXT → plain text
       ▼              DOCX/PPTX → python-docx/python-pptx
┌──────────────┐
│ Normalize    │ ← Strip formatting, unify whitespace,
│ & Clean      │     handle encoding, remove boilerplate
└──────┬───────┘
       ▼
┌──────────────┐
│ Chunking     │ ← Split into passages (app/ingestion/chunking/)
│ (splitter)   │     Paragraph boundary aware, configurable size
└──────┬───────┘
       ▼
┌──────────────┐
│ Embedding    │ ← Convert each chunk to a vector
│ (Jina API /  │     app/services/retrieval/embedding.py
│  fallback)   │
└──────┬───────┘
       ▼
┌──────────────┐
│ Index (Qdrant)│ ← Upsert vectors + payload into Qdrant
│ Vector Store  │     app/ingestion/processor.py → qdrant_client.upsert()
└──────────────┘
```

## 2. Other Approaches

| Approach | Description | When it works | When it doesn't |
|---|---|---|---|
| **On-the-fly retrieval** | Parse files at query time instead of pre-processing | Truly dynamic content; very small corpora | High latency per query; parsing PDFs repeatedly is wasteful |
| **Whole-document indexing** | Store entire documents, rely on the LLM's context window to find answers | Tiny corpus (<10 docs) | Poor recall for specific questions; wastes tokens on irrelevant content |
| **Heavy ETL pipeline** | Add OCR, entity extraction, summarization, knowledge graph construction | Compliance-heavy environments; search over scanned docs | Overkill for most RAG; introduces failure points; expensive to maintain |
| **Raw file dump + BM25** | No chunking; index files directly with keyword search | Keyword-focused queries; simple setup | No semantic matching; poor for paraphrased questions |

## 3. Why This Approach Was Chosen

**Why preprocessing + chunking + embedding:**
- **Separation of concerns.** Ingestion runs offline (batch) and produces artifacts that the online retrieval path can read instantly. Query-time latency is not affected by document parsing complexity.
- **Fine-grained attribution.** Chunk-level indexing means we can tell the user exactly which paragraph or section contains the answer. Whole-document indexing cannot do this.
- **Stable embeddings.** Pre-computing embeddings means the query path only embeds the query (one network call) rather than re-embedding all documents.
- **Reproducibility.** The `processed_data/` directory stores intermediate JSON artifacts so you can debug, re-index, or switch vector stores without re-parsing and re-chunking.

**Why not heavy ETL:**
- The documents in this project are already well-structured technical content (Kubernetes docs, Intel hardware specs). They don't need OCR, entity linking, or knowledge graph extraction. Adding those stages would increase maintenance burden without proportional benefit.

**Why not on-the-fly:**
- Parsing a 100-page PDF at query time would take seconds — unacceptable for a production API targeting <2s response times.

## 4. How Ingestion Is Implemented in This Repo

### Entry point

`app/ingestion/processor.py` contains the orchestration. The main functions are:

- `process_file(file_path, filename, source_type)` — processes a single file through the full pipeline.
- `process_directory(dir_path, source_type)` — iterates over all files in a directory.
- `run_universal_ingestion(base_dir, explicit_source_type, wipe)` — top-level entry for bulk indexing.

### File-type detection and loading

The loader is selected based on the file extension:

```python
ext = filename.lower().rsplit(".", 1)[-1]
if ext == "pdf":
    full_text = parse_pdf(file_path)
elif ext in ("html", "htm"):
    full_text = parse_html(file_path)
elif ext == "txt":
    full_text = parse_text(file_path)
elif ext in ("docx", "pptx"):
    full_text = parse_office(file_path)
```

Each parser lives in `app/ingestion/loaders/`:
- `html.py` — uses BeautifulSoup or lxml to extract readable text from HTML.
- `pdf.py` — uses pdfplumber (preferred for tables) or pypdf for simpler PDFs.
- `text.py` — plain file read with encoding detection.
- `office.py` — python-docx (DOCX) and python-pptx (PPTX) extraction.

### Chunking

The chunker is a paragraph-aware splitter in `app/ingestion/chunking/splitter.py`:

```python
def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) < chunk_size:
            current_chunk += p + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return [c for c in chunks if c.strip()]
```

Key characteristics:
- **Paragraph boundary aware** — splits on `\n\n` rather than at arbitrary character positions, preserving semantic units.
- **Default chunk size 1500 characters** (not tokens) — this was chosen empirically for the document corpus. In production you'd measure in tokens using a tokenizer.
- **No explicit overlap** — overlap is implicit because adjacent paragraphs may span chunks. A sliding window with explicit overlap would improve recall for cross-paragraph answers.

### Embedding and indexing

After chunking, each chunk is embedded via `app/services/retrieval/embedding.py`:

```python
embeddings = embed_texts(chunks)
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
qdrant_client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
```

The `payload` field in Qdrant stores the original text and source metadata alongside the vector. This is critical because the retriever returns both the vector match and the human-readable content.

### Processed data artifacts

Before embedding, the processor writes a JSON file to `processed_data/<source_type>/<filename>.json`:

```python
processed_data = {
    "filename": filename,
    "source_type": source_type,
    "chunks": chunks,
}
```

This serves as a local cache: if the vector DB is rebuilt, you don't need to re-parse and re-chunk — just re-embed from the cached chunks.

## 5. How It Helps Our Project

- **Reproducible indexing.** Running the processor twice on the same input produces identical intermediate artifacts (though Qdrant point IDs will differ). This makes debugging straightforward.
- **Multi-format support.** The project ingests PDFs, HTML, TXT, DOCX, and PPTX — covering the range of enterprise documentation formats.
- **Observability.** Every step is wrapped in `logfire.span()` so you can trace ingestion latency per file and diagnose failures.
- **Graceful failure.** If one file in a batch fails, processing continues with the remaining files. The error is logged and the system moves on.

## 6. What Could Be Better

- **Incremental / watch mode.** Currently, the processor re-processes all files every time. A file watcher that detects new or modified files and indexes only the delta would be much faster for ongoing ingestion.
- **Token-aware chunking.** The current chunker uses character counts. Switching to token counts (via a tokenizer like `tiktoken`) would give more predictable prompt budget behavior.
- **Explicit overlap.** A sliding window with 10-20% overlap ensures that answers spanning chunk boundaries aren't missed.
- **OCR for scanned PDFs.** Some PDFs contain images of text rather than selectable text. Integrating Tesseract or Azure Document Intelligence would handle those cases.
- **Deduplication.** If the same document appears in multiple source directories, the system would index duplicate chunks. Content hash comparison could skip duplicates.
- **Metadata enrichment.** Adding document titles, section headings, and page numbers to each chunk's payload would improve the LLM's ability to cite sources.

## 7. Experience Notes (Building from Scratch)

- **Start with one file type.** Get the pipeline working end-to-end with a single TXT file before adding PDF, HTML, DOCX parsers. PDF parsing is surprisingly tricky (tables, columns, images) and can consume days of debugging.
- **Validate extracted text.** After writing each loader, manually inspect 5-10 files to check for:
  - Garbled characters (encoding issues)
  - Missing text (PDFs with non-standard fonts)
  - Duplicate content (page headers/footers being extracted as content)
  - Extraneous boilerplate (navigation bars in HTML)
- **Test chunk boundaries.** For each document type, print the first few chunks and verify that paragraphs aren't split mid-sentence and that chunk boundaries make semantic sense.
- **Log extraction warnings.** When a PDF fails to parse, log the error and continue. A single corrupt file should not block the entire index.
- **Plan for re-indexing.** The `run_universal_ingestion` function supports a `wipe=True` parameter that clears the Qdrant collection before re-indexing. Use this when changing embedding models or chunk size so you don't have mixed-version data.
- **Processed data as audit trail.** The JSON files in `processed_data/` are your ground truth for what was indexed. If the vector DB gets corrupted, you can rebuild from these files without re-parsing the originals.
- **Monitor embedding API costs.** Jina Embeddings API charges per token. During heavy re-indexing, embedding costs can add up quickly. Consider batching (the code uses `BATCH_SIZE=64`) and caching embeddings for unchanged content.
