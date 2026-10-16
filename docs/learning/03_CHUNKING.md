# Chunking — Splitting Documents into Retrieval Passages

## 1. What It Is

Chunking is the process of dividing a long document into smaller, self-contained passages (chunks) that serve as the atomic retrieval units in a RAG system. Instead of retrieving entire documents, the system retrieves individual chunks, which are then passed to the LLM as context.

**Why chunking matters:**
- **Precision.** A single chunk about "SR-IOV configuration" is more useful than a 50-page Intel NIC manual.
- **Token budget.** LLMs have finite context windows (8K-128K tokens). Small chunks let you fit more evidence into the prompt.
- **Attribution.** If each chunk has metadata (source, page number), the LLM can cite exactly where the answer came from.
- **Retrieval quality.** ANN search works best when vectors represent a single coherent concept. Mixing multiple topics in one chunk confuses the similarity measurement.

**The fundamental tradeoff:**
- **Small chunks (128-256 tokens):** High precision, easy attribution, but may lack surrounding context needed to understand the answer. More chunks to search = slower retrieval.
- **Large chunks (512-1024 tokens):** More context, better for answers that span paragraphs, but less precise; a chunk might contain both relevant and irrelevant content.
- **Whole documents:** Maximum context, but terrible precision and attribution; the LLM must search inside the document itself.

## 2. Other Approaches

| Strategy | How it works | Pros | Cons |
|---|---|---|---|
| **Fixed-size (character)** | Split every N characters | Simple; predictable chunk count | Splits mid-word/sentence; no semantic boundaries |
| **Fixed-size (token)** | Split every N tokens (via tokenizer) | Predictable token budget; no mid-word splits | Still splits mid-sentence; requires tokenizer |
| **Paragraph-based** | Split on `\n\n` (blank line) | Preserves semantic units; natural boundaries | Uneven chunk sizes; very long paragraphs still exceed limits |
| **Recursive character split** | Try paragraph → sentence → character | Best-effort semantic preservation | Complex; may produce very small chunks for edge cases |
| **Semantic chunking** | Use embeddings to detect topic boundaries | Best semantic coherence | Computationally expensive; needs model inference per document |
| **Late chunking / ColBERT** | Keep document-level embeddings, score token-level | No chunking needed; high precision | Complex; special retrieval setup needed |
| **Sliding window** | Overlapping chunks (e.g., stride=256, chunk=512) | No information lost at boundaries | Duplicate content; more storage; potential redundant retrieval |

## 3. Why This Approach Was Chosen

**Why paragraph-aware chunking at ~1500 characters:**
- **Paragraph boundaries preserve meaning.** Splitting at `\n\n` keeps coherent sections together. Technical documentation is written in paragraphs; breaking mid-paragraph loses context.
- **~1500 characters ≈ ~350-400 tokens** (for English technical text). This is a moderate size: large enough to provide sufficient context for most answers, small enough to fit 5 chunks in a 4K token window.
- **Simplicity.** A paragraph splitter is easy to understand, debug, and tune. It has no external dependencies and runs in microseconds per document.

**Why not semantic chunking:**
- Semantic chunking (using embeddings to detect topic shifts) is more accurate but requires processing each document through an embedding model just to determine chunk boundaries. For a project that prioritizes quick iteration, the added complexity wasn't justified.

**Why not simple fixed-size:**
- Fixed-size splitting would cut mid-sentence or mid-paragraph ("To configure SR-IOV, you must enable\n\nVirtualization Technology in th"), producing chunks that are confusing to both the retriever and the LLM.

## 4. How Chunking Is Implemented in This Repo

### The chunker

`app/ingestion/chunking/splitter.py`:

```python
def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """
    Simple semantic-ish chunker that splits by paragraphs.
    Ensures chunks do not exceed the specified size.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

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

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks
```

**How it works step by step:**
1. Split the full text on `\n\n` (double newline = paragraph boundary).
2. Accumulate paragraphs into `current_chunk`.
3. If adding the next paragraph would exceed `chunk_size`, flush `current_chunk` and start a new one with the current paragraph.
4. After the loop, flush any remaining text.
5. Filter out empty chunks.

**Edge cases handled:**
- **Empty text** → returns `[]`.
- **Single long paragraph** → becomes its own chunk (even if it exceeds `chunk_size`). The chunker does not sub-split paragraphs.
- **Many short paragraphs** → efficiently merged into chunks up to `chunk_size`.
- **Trailing whitespace** → stripped via `.strip()`.

**Edge cases NOT handled (known limitations):**
- A paragraph longer than `chunk_size` (e.g., a 5000-character bullet list) will become a single oversize chunk.
- No minimum chunk size — if a document ends with a one-line paragraph, it becomes a tiny chunk.
- No overlap — adjacent chunks share no tokens, so an answer that spans the boundary between two chunks might be missed.

### Integration in the ingestion pipeline

In `app/ingestion/processor.py`:

```python
full_text = parse_pdf(file_path)  # or parse_html, parse_text, etc.
chunks = chunk_text(full_text)
embeddings = embed_texts(chunks)
```

The chunking step sits between text extraction and embedding. If you change `chunk_size`, you must re-run ingestion to produce new chunks and embeddings.

## 5. How It Helps Our Project

- **Balanced chunk size.** 1500 characters provides enough context for most technical answers (a paragraph or two of documentation) while fitting comfortably within the LLM's context window alongside other chunks.
- **Semantic preservation.** Because chunking respects paragraph boundaries, each chunk is a coherent unit of technical documentation. The retriever and LLM see complete thoughts, not truncated sentences.
- **Deterministic.** The same input text always produces the same chunks, making debugging and testing predictable.
- **Lightweight.** The splitter runs in microseconds per document — no API calls, no GPU needed.

## 6. What Could Be Better

- **Token-aware chunking.** The current chunker uses character counts. Switching to token counts (via `tiktoken` or the model's tokenizer) would ensure each chunk fits exactly within the LLM's context window without trial-and-error.
- **Explicit overlap.** A sliding window with 10-20% overlap (e.g., last 100 characters of chunk N duplicated at start of chunk N+1) would prevent information loss at chunk boundaries.
- **Recursive splitting for long paragraphs.** If a single paragraph exceeds `chunk_size`, recursively split it by sentences (then by clauses) to avoid oversize chunks.
- **Minimum chunk size.** Discard or merge chunks shorter than ~50 tokens. Tiny chunks waste index space and rarely provide useful context.
- **Heading-aware chunking.** For HTML and DOCX documents, include the document heading hierarchy (H1/H2/H3) as context for each chunk. This helps the LLM understand where the chunk fits in the document structure.
- **Section metadata.** Store the section title and position with each chunk payload so the LLM can cite "from section 3.2: SR-IOV Configuration."

## 7. Experience Notes (Building from Scratch)

- **Test chunk sizes empirically.** Create a set of 20 test questions, manually annotate the correct answer passage for each, then test different chunk sizes (256, 512, 768, 1024, 1500 characters) and measure recall@5. The optimal size depends on your document structure and question style.
- **Inspect chunk boundaries manually.** After running the chunker on a few documents, print the first and last sentence of each chunk. If you see sentences cut off mid-way, your chunker needs improvement.
- **Beware of boilerplate.** HTML documents often have navigation bars, sidebars, and footers that become chunks. Clean the extracted text before chunking (e.g., remove elements by CSS class).
- **Chunk for the LLM, not just the retriever.** A chunk that is perfect for retrieval might be useless to the LLM if it lacks context. Consider prepending the document title or section heading to each chunk.
- **Version your chunking strategy.** Store the chunking parameters (method, size, overlap) in the processed data metadata. When you change strategies, you can identify which chunks need re-indexing.
- **Be careful with code blocks.** Technical documentation contains code blocks. Try to keep code blocks intact within a single chunk — splitting `kubectl apply -f deployment.yaml` across two chunks would lose meaning.
- **Overlap is cheap.** Doubling your chunk count (50% overlap) doubles storage but can significantly improve recall for cross-boundary answers. The cost in Qdrant is small; the benefit in answer quality can be large.
