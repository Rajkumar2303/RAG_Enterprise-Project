# Agents — Planner, Retriever, Responder Architecture

## 1. What It Is

An agent is a program that uses an LLM to decide which actions to take and in what order to accomplish a task. In this project, the agent is a LangGraph state machine with three nodes: the Planner decomposes the user request, the Retriever fetches relevant context, and the Responder synthesises the final answer.

**The agent lifecycle for a technical query:**

```
User: "How do I configure SR-IOV on an Intel E810?"
    │
    ▼
┌──────────────────────────────────┐
│ 1. Planner                       │
│    Intent: "Technical"           │
│    Search query: "SR-IOV config  │
│    Intel E810 virtual functions" │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│ 2. Retriever                     │
│    Qdrant search (15 cand.)      │
│    Reranker (top-5)              │
│    Returns: [doc1, doc2, ...]    │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│ 3. Responder                     │
│    Assembles prompt:             │
│      CONTEXT: doc1..doc5         │
│      HISTORY: ...                │
│      QUESTION: ...               │
│    Calls LLM → final answer      │
└──────────────────────────────────┘
```

**For conversational queries, the agent skips retrieval entirely:**

```
User: "Hi, what can you help me with?"
    │
    ▼
┌──────────────────────────────────┐
│ 1. Planner                       │
│    Intent: "CONVERSATIONAL"      │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│ 2. Responder (no retrieval)      │
│    Uses conversation history     │
│    Calls LLM → greeting          │
└──────────────────────────────────┘
```

## 2. Other Approaches

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **Monolithic prompt** | Single LLM call with all context | Simple; low latency | No retrieval; limited context; no multi-step reasoning |
| **Chain (sequential)** | Fixed pipeline: A → B → C | Predictable; easy to test | Cannot adapt to the input; same steps every time |
| **Router** | Classify query → dispatch to specialized handler | Good for distinct domains | Limited to predefined routes; no dynamic planning |
| **ReAct agent** | LLM reasons + acts in a loop | Very flexible; can use many tools | Expensive (many LLM calls); unpredictable latency; complex to debug |
| **Plan-and-execute** | Plan first, then execute each step | Structured; controllable | Planner quality determines everything; slow for simple queries |
| **LangGraph (this repo)** | State graph with conditional edges | Deterministic routing; easy to understand; testable | Less flexible than ReAct; needs explicit node logic |

## 3. Why This Architecture Was Chosen

**Why LangGraph with Planner → Retriever → Responder:**

- **Separation of concerns.** Each node has a single responsibility: Planner decides *what* to do, Retriever fetches *evidence*, Responder produces *output*. If one step fails, you know exactly where.
- **Conditional routing.** The Planner dynamically decides whether to retrieve (technical query) or skip (conversational). This avoids wasting time and tokens on retrieval for simple greetings.
- **State management.** LangGraph maintains a typed `AgentState` that passes data between nodes. Messages accumulate (not overwrite), so the Responder sees the full conversation history.
- **Checkpointing.** The graph supports a Postgres-backed checkpointer (`PostgresSaver`) so the agent can resume after a crash. MemorySaver is used for dev; Postgres for production.
- **Testability.** Because the graph is a directed acyclic graph with well-defined inputs/outputs, you can test each node in isolation and integration-test the full graph with mock data.

**Why not ReAct:**
- ReAct agents are unstructured — the LLM decides each step dynamically. This makes them powerful but unpredictable in latency and cost. For a production RAG system that needs consistent response times, a structured graph is safer.

**Why not monolithic prompt:**
- A single prompt cannot handle both conversational greetings and technical multi-document synthesis. The Planner's conditional routing adapts the pipeline to the query type automatically.

## 4. How the Agent Is Implemented in This Repo

### State definition

`app/agents/state.py`:

```python
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]  # accumulated conversation
    current_query: str                                # CONVERSATIONAL or refined search query
    documents: List[str]                              # retrieved passages
    plan: List[str]                                   # step-by-step plan for UI display
    status: str                                       # human-readable status
    final_answer: str                                 # LLM-generated answer
```

The `Annotated[list, operator.add]` annotation tells LangGraph to append (not replace) messages, enabling multi-turn conversation history.

### Graph construction

`app/agents/graph.py`:

```python
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)

workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", route_planner, {
    "retriever": "retriever",
    "responder": "responder",
})
workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)

checkpointer = create_checkpointer()  # Postgres or MemorySaver
app = workflow.compile(checkpointer=checkpointer)
```

### Planner node

`app/agents/nodes/planner.py`:

```python
def planner_node(state: AgentState):
    history = ...   # compile conversation history
    prompt = f"""
    Analyze the conversation history and the latest user message.
    
    Task:
    1. If greeting or answerable from history → output 'CONVERSATIONAL'
    2. If technical question → output a refined search query
    
    Output ONLY 'CONVERSATIONAL' or the search query.
    """
    llm = _get_llm()  # Portkey/DeepSeek, or None for local fallback
    decision = llm.invoke(prompt).content.strip()
    
    if decision == "CONVERSATIONAL":
        return {"current_query": "CONVERSATIONAL", "status": "Conversational", ...}
    return {"current_query": decision, "status": f"Searching for: {decision}", ...}
```

Key details:
- **Local fallback:** If no LLM is configured, the planner defaults to `CONVERSATIONAL` to avoid hard failures in development.
- **Prompt design:** The prompt explicitly lists the two output formats and instructs the LLM to output *only* the decision (no extra text).

### Retriever node

`app/agents/nodes/retriever.py`:

```python
def retrieve_node(state: AgentState):
    query = state["current_query"]
    raw_results = search_enterprise_knowledge(query, limit=15)
    doc_contents = [doc["content"] for doc in raw_results]
    reranked_contents = rerank_documents(query, doc_contents, top_n=5)
    formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]
    return {"documents": formatted_docs, "status": "Found technical context.", ...}
```

### Responder node

`app/agents/nodes/responder.py`:

```python
def generate_node(state: AgentState):
    if query == "CONVERSATIONAL":
        prompt = f"Use CONVERSATION HISTORY to answer: {user_msg}"
    else:
        prompt = f"""
        You are a Senior Technical Architect.
        Use TECHNICAL CONTEXT + HISTORY to answer: {user_msg}
        CONTEXT: {full_context}
        """
    response = _generate_response(prompt)  # with tenacity retry
    return {"final_answer": content, ...}
```

Key details:
- **Two prompt templates** — one for conversational, one for technical RAG.
- **Context truncation** — limits context to `max_context_chars=25000` to stay within token limits.
- **Cache tracking** — extracts `x-portkey-cache-status` to display cache hits in the UI.

### Routing logic

```python
def route_planner(state: AgentState):
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    return "retriever"
```

This is a simple string comparison — no LLM call needed for routing.

### Checkpointer (persistence)

`app/agents/graph.py` `create_checkpointer()`:

```python
try:
    pool = ConnectionPool(conninfo=settings.postgres_uri, max_size=20)
    checkpointer = PostgresSaver(pool)
except Exception as e:
    checkpointer = MemorySaver()  # dev fallback
```

Postgres persistence means agent state survives server restarts. Each user conversation has a `thread_id` that maps to its checkpoint.

## 5. How It Helps Our Project

- **Intent-aware routing.** The Planner separates simple conversations from technical queries, avoiding unnecessary retrieval latency for greetings and reducing LLM costs.
- **Multi-turn conversations.** The accumulated messages list lets users ask follow-up questions ("what about SR-IOV on the E810?" after "tell me about Intel NICs").
- **Observable intermediate state.** The `plan` and `status` fields track every decision, making it easy to debug bad answers.
- **Production-ready persistence.** Postgres checkpointing ensures that in-flight conversations survive server restarts.
- **Cache acceleration.** The Responder tracks Portkey cache status, and the UI can display "Cache: Hit ⚡" to show when an answer was served from cache.

## 6. What Could Be Better

- **Sub-query decomposition.** For complex questions ("compare SR-IOV and DPDK"), the Planner could generate multiple sub-queries, each retrieving different documents, then merge results.
- **Retry per node.** Currently only the Responder has retry logic. The Retriever and Planner could also benefit from retry with fallback strategies.
- **Node-level caching.** If the same query arrives twice, the Retriever and Planner produce the same output. Caching their outputs (keyed by query) would save time and money.
- **Graph visualization.** LangGraph can render the graph as a Mermaid diagram. Exposing this in a debug endpoint would help visualize the agent's decision path.
- **Streaming.** The Responder currently waits for the full LLM response. Streaming tokens to the client would improve perceived latency.
- **Tool-use expansion.** The agent could be extended with more tools: calculator, code interpreter, web search, etc. The Planner would decide which tool to invoke.

## 7. Experience Notes (Building from Scratch)

- **Start without agents.** Build a simple query → retrieve → answer pipeline first. Only add the Planner when you need to distinguish conversational from technical queries. Premature agent complexity is a common mistake.
- **Test each node in isolation.** Write unit tests for the Planner (mock the LLM), Retriever (mock Qdrant), and Responder (mock the LLM). This catches integration bugs early.
- **Log the Planner's decision.** Every time the planner runs, log the input query and output decision. This helps you tune the prompt when the planner misclassifies queries.
- **Use LangGraph Studio for debugging.** LangGraph has a visual debugger that shows the graph state at each step. It's invaluable for understanding unexpected behavior.
- **Design state carefully.** The `AgentState` TypedDict is the contract between nodes. Adding a new field later requires updating all nodes. Plan for fields like `documents`, `plan`, and `status` from the start.
- **Handle node failures gracefully.** If the Retriever throws an exception (Qdrant down), the Responder should still produce an answer ("I was unable to search the knowledge base. Here's what I know: ...") rather than crashing the whole conversation.
- **Set up Postgres early.** MemorySaver loses state on restart. Deploy with PostgresSaver from the first production release to avoid breaking in-flight conversations.
