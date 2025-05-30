import streamlit as st
import numpy as np
import faiss
import json
import openai
import tiktoken
import pandas as pd
from PIL import Image
from collections import defaultdict
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re

# ====== Streamlit Config ======
st.set_page_config(page_title="ESGenie – Custom RFP Bot", layout="wide")

# ====== Constants ======
BASE_DOC_URL = "https://yourdomain.com/docs/"  # Update to your hosting location
SHEET_NAME = "ESGenieChatLogs"

# ====== Session State ======
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query" not in st.session_state:
    st.session_state.query = ""

# ====== Hero Section ======
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image("logo.png", width=120)
st.markdown("<h1 style='margin-bottom: 0;'>ESGenie</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px;'>Generate ESG-related RFP responses with citations and document links</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")

# ====== Suggestion Buttons ======
st.markdown("### 💡 Try a prompt:")

col1, col2, col3,col4, col5 = st.columns([1, 1, 1, 1, 1])  # Equal width, minimal spacing

with col1:
    st.button("What are Bain's sustainability commitments?", key="prompt1")

with col2:
    st.button("What are the Bain's waste diversion efforts?", key="prompt2")

with col3:
    st.button("What are the Bain's ISO certifications, if any?", key="prompt3")

# Handle button behavior outside to avoid space/logic issues
if st.session_state.get("prompt1"):
    st.session_state.query = "What are Bain's sustainability commitments?"
elif st.session_state.get("prompt2"):
    st.session_state.query = "What are Bain's waste diversion policy?"
elif st.session_state.get("prompt3"):
    st.session_state.query = "What are Bain's ISO certifications, if any?"


# ====== Configuration ======
openai.api_key = st.secrets["openai_api_key"]
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4"
TOP_K = 20
MAX_CONTEXT_TOKENS = 6000

# ====== Document Priorities ======
DOCUMENT_PRIORITIES = {
    "GRI report 2023.pdf": 1,
    "Environmental-policy.pdf": 2,
    "climate-transition-plan-2024.pdf": 3,
    "Bain DEI report.pdf": 4,
    "2023_carbon-credit-disclosure.pdf": 5,
    "bain-wef-and-tcfd-report-2023.pdf": 6,
    "Bain Overview FAQ.pdf": 7,
    "Full RFP FAQ Export.pdf": 8,
    "Client RFP deck.pdf": 9,
    "Global Safety & Security FAQ.pdf": 10,
    "bain-sustainable-procurement-policy.pdf": 11,
    "Professional Standards FAQ.pdf": 12,
    "Social Impact.pdf": 13,
    "human-rights-statement-05.2024.pdf": 14,
    "sustainable-procurement-factsheet-v3-05.02.2024.pdf": 15,
    "Diversity and Inclusion FAQ.pdf": 16
}

# ====== Google Sheets Logger ======
def log_chat_to_gsheet(question, answer, sources):
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).sheet1

        sources_text = ", ".join([
            f"{chunk['metadata'].get('document', 'Unknown')} (page {chunk['metadata'].get('page', '?')})"
            for chunk in sources
        ])
        sheet.append_row([
            pd.Timestamp.now().isoformat(),
            question,
            answer or "No answer generated.",
            sources_text
        ])
    except Exception as e:
        st.error(f"Logging to Google Sheets failed: {e}")

# ====== Load Vector DB & Metadata ======
@st.cache_resource
def load_resources():
    index = faiss.read_index("4.my_vector_db.index")
    with open("5.metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# ====== Embedding Function ======
def get_embedding(text):
    try:
        response = openai.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return np.array(response.data[0].embedding, dtype="float32")
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

# ====== Chunk Search ======
def search_chunks(query_text, index, metadata):
    ids = [item["id"] for item in metadata]
    texts = [item["text"] for item in metadata]
    meta_lookup = {item["id"]: item["metadata"] for item in metadata}

    query_vec = get_embedding(query_text)
    if query_vec is None:
        return []

    D, I = index.search(np.array([query_vec]), TOP_K)
    results = []

    for rank, i in enumerate(I[0]):
        if i == -1:
            continue
        chunk_id = ids[i]
        text = texts[i]
        meta = meta_lookup.get(chunk_id, {})
        doc = meta.get("document", "Unknown")
        page = meta.get("page", "?")
        priority = DOCUMENT_PRIORITIES.get(doc, 1000)
        score = float(D[0][rank]) * (1 + priority / 100)
        results.append({
            "text": text,
            "score": score,
            "metadata": meta
        })

    return sorted(results, key=lambda x: x["score"])

# ====== GPT Answer Generator with Inline Source Tags ======
def generate_answer(query, context_chunks):
    encoding = tiktoken.encoding_for_model("gpt-4")
    context_parts = []
    total_tokens = 0

    for chunk in context_chunks:
        doc = chunk["metadata"].get("document", "Unknown")
        page = chunk["metadata"].get("page", "?")
        tagged = f"{chunk['text']}\n(Source: {doc}, page {page})"
        token_count = len(encoding.encode(tagged))
        if total_tokens + token_count > MAX_CONTEXT_TOKENS:
            break
        context_parts.append(tagged)
        total_tokens += token_count

    context_text = "\n\n".join(context_parts)

    try:
        response = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a concise expert assistant. Answer only using the provided context. "
                    "Include inline citations like (Source: filename, page X)."
                )},
                {"role": "user", "content": f"{context_text}\n\nQuestion: {query}"}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip(), context_chunks
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None, []

# ====== Clear Chat Button ======
if st.button("🗑️ Clear Conversation History"):
    st.session_state.chat_history = []
    st.session_state.query = ""

# ====== User Input ======
query = st.text_input("🔍 Ask your question:", key="query")

# ====== Process Query ======
if query.strip():
    with st.spinner("Processing..."):
        index, metadata = load_resources()
        top_chunks = search_chunks(query, index, metadata)
        answer, used_chunks = generate_answer(query, top_chunks) if top_chunks else ("No relevant context found.", [])
        st.session_state.chat_history.append({
            "question": query,
            "answer": answer,
            "sources": used_chunks
        })
        log_chat_to_gsheet(query, answer, used_chunks)

# ====== Display History ======
for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
    st.markdown(f"### 🧠 Question {len(st.session_state.chat_history) - i + 1}")
    st.markdown(chat["question"])
    st.markdown("### 💡 Answer")
    inline = re.sub(r'\(Source: (.*?), page (\d+)\)', r'📝 [\1 – p.\2]', chat["answer"])
    st.markdown(inline, unsafe_allow_html=True)

    st.markdown("### 📚 Sources")
    grouped = defaultdict(list)
    for chunk in chat["sources"]:
        meta = chunk["metadata"]
        doc = meta.get("document", "Unknown")
        page = meta.get("page", "?")
        grouped[doc].append(page)

    for doc, pages in grouped.items():
        doc_url = f"{BASE_DOC_URL}{doc.replace(' ', '%20')}"
        page_str = ", ".join(map(str, sorted(set(pages))))
        st.markdown(f"- [**{doc}** (pages {page_str})]({doc_url})")

    st.markdown("---")
