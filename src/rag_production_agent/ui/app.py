"""Streamlit frontend. Talks to the FastAPI service only (no direct Inngest/Qdrant access)."""

import hashlib
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="RAG over your PDFs", page_icon="📄", layout="centered")

# Streamlit reruns this script on every interaction, and the uploaded file stays
# set across reruns. Track what was already sent so each file is ingested once.
if "ingested" not in st.session_state:
    st.session_state.ingested = set()

st.title("Upload a PDF")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    data = uploaded.getvalue()
    key = hashlib.sha256(data).hexdigest()
    if key not in st.session_state.ingested:
        with st.spinner("Uploading and starting ingestion..."):
            resp = requests.post(
                f"{API_URL}/ingest",
                files={"file": (uploaded.name, data, "application/pdf")},
                timeout=60,
            )
        if resp.ok:
            st.session_state.ingested.add(key)
            st.success(f"Ingestion started for **{uploaded.name}**. Large files can take a few minutes.")
        else:
            st.error(f"Upload failed ({resp.status_code}): {resp.text}")
    else:
        st.info(f"Already ingested: {uploaded.name}")

st.divider()
st.title("Ask a question about your PDFs")

with st.form("query"):
    question = st.text_input("Your question")
    top_k = st.number_input("Chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Retrieving and generating..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question.strip(), "top_k": int(top_k)},
                timeout=180,
            )
        except requests.RequestException as e:
            st.error(f"Can't reach the API at {API_URL}: {e}")
            st.stop()
    if not resp.ok:
        # Show the API's own explanation (e.g. "Vector store unreachable…") rather than a bare status code
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        st.error(f"Query failed ({resp.status_code}): {detail}")
        st.stop()
    result = resp.json()

    st.subheader("Answer")
    st.write(result.get("answer") or "(No answer)")
    if sources := result.get("sources"):
        st.caption("Sources")
        for s in sources:
            st.write(f"- {s}")
