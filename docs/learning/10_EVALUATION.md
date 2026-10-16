# Evaluation — Measuring RAG Quality and Performance

## 1. What It Is

Evaluation is the systematic process of measuring how well a RAG system performs across multiple dimensions: retrieval accuracy, answer correctness, safety guardrail effectiveness, and user experience. Without evaluation, improvements to chunking, embedding, prompt templates, or model choices are guesswork.

**Dimensions of evaluation in this project:**

| Dimension | What it measures | How measured |
|---|---|---|
| **Retrieval quality** | Are the right passages being found? | Recall@k, precision@k, MRR |
| **Answer factuality** | Is the LLM answer correct and grounded? | RAGAS metrics (faithfulness, answer relevancy) |
| **Guardrail effectiveness** | Are unsafe inputs caught? | True positive rate, false positive rate |
| **End-to-end accuracy** | Does the system produce the right answer? | Exact match, semantic similarity to golden answer |
| **Latency** | How fast is each step? | Per-step timing via logfire spans |

## 2. Other Approaches

| Approach | What it measures | Pros | Cons |
|---|---|---|---|
| **BLEU / ROUGE** | N-gram overlap with reference answer | Simple; well-known | Poor correlation with human judgment; penalizes valid paraphrasing |
| **BERTScore** | BERT embedding similarity to reference | Better than BLEU/ROUGE | Still requires reference answer; expensive to compute |
| **RAGAS (this repo)** | Faithfulness + answer relevancy + context precision | Specifically designed for RAG; no reference answer needed for some metrics | Requires an LLM judge; adds cost |
| **LLM-as-judge (GPT-4)** | Ask an LLM to rate answer quality | Most flexible; can catch nuanced errors | Expensive; slow; judge LLM has biases |
| **Human evaluation** | Manual rating by domain experts | Gold standard | Very expensive; slow; not scalable |
| **A/B testing** | Compare two system versions with live users | Real-world signal | Requires traffic; hard to isolate variables |

## 3. Why This Approach Was Chosen

**Why a multi-metric approach with RAGAS + guardrail eval:**
- **RAGAS for answer quality.** RAGAS (Retrieval Augmented Generation Assessment) provides metrics specifically designed for RAG: `faithfulness` (is the answer grounded in the retrieved context?) and `answer_relevancy` (how relevant is the answer to the question?). These don't require a reference answer, making them practical for continuous evaluation.
- **Guardrail eval for safety.** Separate from answer quality, we measure whether guardrails correctly block unsafe inputs (true positives) and don't block safe inputs (false positives).
- **Golden dataset for end-to-end.** A curated set of question-answer pairs lets us measure exact-match and semantic similarity, providing a concrete "score" for each system version.

**Why not human-only evaluation:**
- Human evaluation is essential for final quality assurance, but too slow and expensive for iterative development. Automated metrics let us iterate quickly and reserve human review for major releases.

**Why not BLEU/ROUGE only:**
- BLEU and ROUGE measure surface-level text overlap, which penalizes valid paraphrasing. A correct answer phrased differently from the reference gets a low BLEU score, even if it's factually accurate.

## 4. How Evaluation Is Implemented in This Repo

### Directory structure

```
evals/
├── app.py                   # Streamlit UI for running evaluations
├── data_parser.py           # Parse evaluation results
├── golden_dataset.json      # Curated Q&A pairs with ground truth
├── og_golden_dataset.json   # Original/backup golden dataset
├── guardrails_eval.py       # Guardrail-specific evaluation
├── metrics.py               # RAGAS metrics computation
├── pipeline.py              # Pipeline orchestration (load, run, score)
└── run_evals.py             # CLI entry point
```

### Golden dataset

`evals/golden_dataset.json` contains structured test cases:

```json
[
  {
    "question": "How do I configure SR-IOV on an Intel E810?",
    "ground_truth": "Enable SR-IOV in the BIOS, then use `echo 64 > /sys/class/net/enp0s1f0np0/device/sriov_numvfs`",
    "source": "intel_e810_sriov_guide.pdf",
    "category": "networking"
  },
  ...
]
```

Each entry has:
- **question** — the user's input.
- **ground_truth** — the expected correct answer (or key facts).
- **source** — which document contains the answer (for traceability).
- **category** — for segmenting results by topic.

### Running the pipeline

`evals/pipeline.py` `run_pipeline()`:

```python
def run_pipeline(golden):
    results = []
    for entry in golden:
        response = query_rag_api(entry["question"])
        results.append({
            "question": entry["question"],
            "ground_truth": entry["ground_truth"],
            "llm_answer": response["answer"],
            "retrieved_docs": response["documents"],
            "latency_ms": response["latency_ms"],
        })
    return results
```

This function sends each golden question through the live RAG API and records the answer, retrieved documents, and latency.

### Computing RAGAS metrics

`evals/metrics.py` `run_all_metrics()`:

```python
def run_all_metrics(enriched_dataset):
    # RAGAS: faithfulness, answer_relevancy, context_precision
    result = evaluate(
        enriched_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    return result.to_pandas()
```

RAGAS uses an LLM judge to score each dimension:
- **Faithfulness:** Does the answer stay true to the retrieved context? (0-1)
- **Answer relevancy:** Is the answer relevant to the question? (0-1)
- **Context precision:** Are the retrieved documents all relevant? (0-1)
- **Context recall:** Did we retrieve all relevant documents? (0-1)

### Guardrail evaluation

`evals/guardrails_eval.py`:

```python
def run_guardrails_eval():
    test_inputs = [
        ("tell me a joke", True),           # should be BLOCKED
        ("hello", True),                     # should be HANDLED
        ("how do I deploy K8s on Intel?", False),  # should PASS
        ...
    ]
    for query, expected_blocked in test_inputs:
        blocked, response = guard(query)
        # Compare against expected_blocked
```

This evaluates whether guardrails correctly identify safe vs. unsafe inputs. Key metrics:
- **True positive rate:** % of unsafe inputs correctly blocked.
- **False positive rate:** % of safe inputs incorrectly blocked.
- **Precision and recall** for the guardrail classifier.

### Streamlit UI

`evals/app.py` provides a 3-tab Streamlit interface:

1. **Tab 1 — Ground Truth:** Review the golden dataset.
2. **Tab 2 — Live Pipeline:** Run the pipeline against the golden dataset and see individual results.
3. **Tab 3 — Eval Metrics:** RAGAS scores with color coding (🟢 ≥0.75, 🟡 0.5-0.75, 🔴 <0.5).

The UI makes it easy to spot regressions: if the faithfulness score drops from 0.90 to 0.70 after a change, you know something went wrong.

## 5. How It Helps Our Project

- **Quantifies quality.** Instead of "this answer feels better," you have concrete scores: faithfulness=0.92, answer_relevancy=0.88. This guides where to invest effort.
- **Detects regressions.** Running the full eval suite before each deployment catches unintended side effects of changes.
- **Guardrail confidence.** The guardrail evaluation proves (with numbers) that the system is safe against known attack patterns.
- **Data-driven tuning.** When retrieval quality is low (context precision <0.7), you know to focus on chunking or embedding improvements. When answer relevancy is low, focus on prompt templates.

## 6. What Could Be Better

- **CI integration.** Make evaluation part of the CI pipeline. Every PR triggers a full eval run, and the PR cannot merge if any metric drops below a threshold.
- **Synthetic data generation.** Use an LLM to generate additional test questions from the document corpus, expanding coverage beyond the manual golden dataset.
- **Latency budgets.** Enforce per-step latency budgets (e.g., retrieval <500ms, guardrail <1s, total <5s) and alert when exceeded.
- **Adversarial test set.** Create test inputs specifically designed to break the system: misspellings, bad grammar, contradictory queries, multi-part questions.
- **Version tracking.** Store evaluation results with git commit hashes so you can track metric changes over time on a dashboard.
- **Calibration.** For RAGAS metrics, determine what scores correlate with acceptable human ratings. Is faithfulness=0.80 good enough, or do we need 0.95?

## 7. Experience Notes (Building from Scratch)

- **Start with 10-20 golden questions.** Don't try to build a comprehensive evaluation suite on day one. Start small, run it manually after every change, and add questions as you discover failure modes.
- **Include edge cases.** Your golden dataset should include:
  - Simple factual questions (one paragraph contains the answer).
  - Multi-document synthesis questions (answer requires combining information from two documents).
  - Out-of-domain questions ("what's the weather?") — should the system refuse or answer based on general knowledge?
  - Ambiguous questions to test the planner's intent detection.
- **Automate the eval run.** A one-command eval (`python -m evals.run_evals`) ensures consistency. Manual copy-paste testing leads to skipped evaluations.
- **Track scores over time.** Keep a CSV or spreadsheet of eval results across versions. A 0.05 drop in faithfulness between versions is a red flag even if the current score is still "acceptable."
- **Use a separate LLM for evaluation.** The judge LLM (used by RAGAS) should be different from the generation LLM to avoid bias. The config supports a `JUDGE_OPENAI_API_KEY` for this purpose.
- **Don't over-optimize for metrics.** Metrics are proxies, not the goal. A system that scores perfectly on faithfulness but produces unhelpful answers is still a bad system. Supplement automated metrics with periodic human review.
- **Record failures, not just scores.** When a question fails (faithfulness <0.5), log the question, retrieved context, and generated answer. This makes debugging straightforward — you can see exactly what went wrong.
