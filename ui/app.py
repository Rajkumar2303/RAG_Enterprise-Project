import os
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv
import pathlib


# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)


# Initialize Logfire
try:
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        print("ERROR: LOGFIRE_TOKEN is empty or None!")
    logfire.configure(token=token)
    # logfire.instrument_requests() # Disabled due to OpenTelemetry bug on Windows: MeterProvider.get_meter() got multiple values for argument 'version'
    LOGFIRE_STATUS = "Connected & Tracing"
except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"
    


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

# --- AVATARS ---
AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")
    
    if st.button("🗑️ Clear History & Memory", width="stretch", type="primary"):
        logfire.warn(f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    # --- File uploader ---
    st.markdown("---")
    st.subheader("📥 Upload Document")
    uploads_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "uploads"))
    pathlib.Path(uploads_dir).mkdir(parents=True, exist_ok=True)

    uploaded_files = st.file_uploader(
        "Upload files to ingest (pdf, txt, html, md, docx, pptx)",
        type=["pdf", "txt", "html", "md", "docx", "pptx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        saved_paths = []
        for uf in uploaded_files:
            save_path = os.path.join(uploads_dir, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(save_path)
        st.success(f"Saved {len(saved_paths)} file(s) to DATA/uploads/")
        for p in saved_paths:
            st.write(p)

        st.info("To ingest these files into the RAG index, run this command in your project root:")
        st.code("python -m app.ingestion.processor DATA/uploads")

# --- MAIN CHAT ---
st.title("🤖 Enterprise Agentic Assistant")


# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your documentation..."):
    # START TRACE: User Interaction
    with logfire.span("💬 User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    # DISTRIBUTED TRACE: Calling Backend
                    with logfire.span("📡 Calling RAG Backend"):
                        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                        url = f"{base_url}/query"
                        payload = {"q": prompt, "thread_id": st.session_state.session_id}
                        # Include Authorization header if RAG_API_KEY is set in .env
                        headers = {}
                        rag_key = os.getenv("RAG_API_KEY")
                        if rag_key:
                            headers = {"Authorization": f"Bearer {rag_key}"}
                        response = requests.post(url, json=payload, timeout=60, headers=headers)
                        data = response.json()

                    # Guardrails can block synchronously without creating a job.
                    if data.get("status") == "Blocked by guardrails.":
                        status.update(label="🛡️ Blocked by guardrails", state="complete", expanded=False)
                        full_answer = data.get("answer", "Blocked by guardrails.")
                    elif "job_id" in data:
                        job_id = data["job_id"]
                        poll_url = f"{base_url}/query/status/{job_id}"
                        result_data = None
                        max_attempts = 60
                        for attempt in range(max_attempts):
                            with logfire.span("🔄 Polling RAG job", job_id=job_id, attempt=attempt):
                                poll_resp = requests.get(poll_url, timeout=30, headers=headers)
                                poll_resp.raise_for_status()
                                poll_data = poll_resp.json()
                            job_status = poll_data.get("status", "UNKNOWN")
                            status.write(f"⏳ Job status: {job_status} (attempt {attempt + 1}/{max_attempts})")
                            if job_status in ("SUCCESS", "FAILURE"):
                                result_data = poll_data.get("result") or poll_data.get("error")
                                break
                            time.sleep(2)
                        if result_data is None:
                            raise RuntimeError("Polling timed out waiting for the RAG job to complete.")
                        if isinstance(result_data, dict):
                            data = result_data
                            status.update(label="✅ Answer Synthesized", state="complete", expanded=False)
                        else:
                            raise RuntimeError(f"RAG job failed: {result_data}")
                    else:
                        # Synchronous response from backend (no job): proceed to render answer
                        status.update(label="✅ Answer Synthesized", state="complete", expanded=False)

                    # Show Reasoning Steps from Backend
                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.write(f"⚙️ {step}")

                    # --- SHOW SOURCES (NESTED EXPANDABLES) ---
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander("📄 View Retrieved Context (Sources)"):
                            for i, source in enumerate(sources):
                                preview = source[:100].replace("\n", " ") + "..."
                                with st.expander(f"Chunk {i+1}: {preview}"):
                                    st.info(source)
                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error(f"Backend Offline or job failed: {e}")
                    st.stop()

            # Final Answer Streaming
            answer_placeholder = st.empty()
            full_answer = data.get("answer", "No response.")

            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)

            answer_placeholder.markdown(full_answer)
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("✅ Chat cycle completed successfully.")
