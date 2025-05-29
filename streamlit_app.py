import streamlit as st
import numpy as np
import faiss
import json
import sqlite3
import pandas as pd
from PIL import Image
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import os

# ====== Streamlit Config ======
st.set_page_config(page_title="ESGenie – Custom RFP Bot", layout="wide")

# ====== Initialize session state for chat history ======
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ====== Branding ======
logo = Image.open("logo.png")
col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=200)
with col2:
    st.markdown("## **ESGenie**")
    st.markdown("Custom RFP & ESG Document Q&A Assistant")

st.markdown("---")

# ====== Configuration ======
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
TOP_K = 20
MAX_CONTEXT_TOKENS = 6000

# ====== Document Priority Mapping ======
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

# ====== Initialize SQLite DB ======
def init_db():
    conn = sqlite3.connect("chat_logs.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            sources TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ====== Save Chat to DB ======
def log_chat_to_db(question, answer, sources):
    source_list = [
        {
            "document": chunk["metadata"].get("document", "Unknown"),
            "page": chunk["metadata"].get("page", "?")
        } for chunk in sources
    ]
    sources_json = json.dumps(source_list)

    conn = sqlite3.connect("chat_logs.db")
    c = conn.cursor()
    c.execute('''
        INSERT INTO chat_history (question, answer, sources)
        VALUES (?, ?, ?)
    ''', (question, answer, sources_json))
    conn.commit()
    conn.close()

# ====== Export Chat Log to Excel ======
def export_chat_log_to_excel():
    conn = sqlite3.connect("chat_logs.db")
    c = conn.cursor()
    c.execute("SELECT question, answer, sources, timestamp FROM chat_history ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()

    data = []
    for q, a, s, t in rows:
        sources = ", ".join([
            f"{src['document']} (page {src['page']})"
            for src in json.loads(s)
        ])
        data.append({
            "Timestamp": t,
            "Question": q,
            "Answer": a,
            "Sources": sources
        })

    df = pd.DataFrame(data)
    df.to_excel("chat_history_log.xlsx", index=False)

# ====== Load resources ======
@st.cache_resource
def load_resources():
    index = faiss.read_index("4.my_vector_db.index")
    with open("5.metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# ====== Embedding Function ======
def get_embedding(text):
    try:
        return EMBEDDING_MODEL.encode(text).astype("float32")
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

# ====== Search Function with Priority Adjustment ======
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
        priority = DOCUMENT_PRIORITIES.get(doc, 1000)
        score = float(D[0][rank])
        adjusted_score = score * (1 + priority / 100)
        results.append({
            "text": text,
            "score": adjusted_score,
            "metadata": meta
        })

    results.sort(key=lambda x: x["score"])
    return results

# ====== Generate Local Answer ======
def generate_answer(query, context_chunks):
    combined_context = "\n\n".join(chunk["text"] for chunk in context_chunks)
    # Only use content from provided documents to generate the response
    if not combined_context.strip():
        return "I'm sorry, I could not find any information relevant to your query in the provided documents.", context_chunks
    else:
        return combined_context[:1000] + "...", context_chunks

# ====== Clear Chat Button ======
if st.button("🗑️ Clear Conversation History"):
    st.session_state.chat_history = []

# ====== App UI ======
query = st.text_input("🔍 Ask your question:")

if query.strip():
    with st.spinner("Processing..."):
        index, metadata = load_resources()
        top_chunks = search_chunks(query, index, metadata)

        if not top_chunks:
            st.warning("No relevant results found.")
        else:
            answer, used_chunks = generate_answer(query, top_chunks)

            if answer:
                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer,
                    "sources": used_chunks
                })
                log_chat_to_db(query, answer, used_chunks)
                export_chat_log_to_excel()

# ====== Display chat history ======
for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
    st.markdown(f"### 🧠 Question {len(st.session_state.chat_history)-i+1}")
    st.markdown(chat["question"])

    st.markdown("### 💡 Answer")
    st.markdown(chat["answer"])

    st.markdown("### 📚 Sources")
    grouped = defaultdict(list)
    for chunk in chat["sources"]:
        meta = chunk["metadata"]
        doc = meta.get("document", "Unknown")
        page = meta.get("page", "?")
        grouped[doc].append(page)

    for doc, pages in grouped.items():
        link = next(
            (c["metadata"].get("link", "#")
             for c in chat["sources"] if c["metadata"].get("document") == doc),
            "#"
        )
        st.markdown(f"- [{doc} (pages {', '.join(map(str, sorted(set(pages))))})]({link})")

    st.markdown("---")
