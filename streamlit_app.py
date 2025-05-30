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
import re

# ====== Streamlit Config ======
st.set_page_config(page_title="ESGenie – Custom RFP Bot", layout="wide")

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
openai.api_key = st.secrets["openai_api_key"]
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4"
TOP_K = 20
MAX_CONTEXT_TOKENS = 6000
BASE_DOC_URL = "https://yourdomain.com/docs/"  # Update this with your document URL base

# ====== Initialize clean chat session ======
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ====== Document Priority ======
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

# ====== Chat Logger ======
def log_chat_to_excel(question, answer, sources):
    file_path = "chat_history_log.xlsx"
    sources_text = ", ".join([
        f"{chunk['metadata'].get('document', 'Unknown')} (page {chunk['metadata'].get('page', '?')})"
        for chunk in sources
    ])
    new_entry = {
        "Timestamp": pd.Timestamp.now().isoformat(),
        "Question": question,
        "Answer": answer,
        "Sources": sources_text
    }
    if os.path.exists(file_path):
        df_existing = pd.read_excel(file_path)
        df_updated = pd.concat([df_existing, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df_updated = pd.DataFrame([new_entry])
    df_updated.to_excel(file_path, index=False)

# ====== Load Vector DB and Metadata ======
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

# ====== Search Chunks ======
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

# ====== GPT Answer Generator (Flexible) ======
def generate_answer(query, context_chunks):
    encoding = tiktoken.encoding_for_model("gpt-4")
    total_tokens = 0
    context_parts = []

    for chunk in context_chunks:
        meta = chunk["metadata"]
        doc = meta.get("document", "Unknown")
        page = meta.get("page", "?")
        chunk_text = f"{chunk['text']}\n(Source: {doc}, page {page})"
        chunk_tokens = len(encoding.encode(chunk_text))
        if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
            break
        context_parts.append(chunk_text)
        total_tokens += chunk_tokens

    context_text = "\n\n".join(context_parts)

    try:
        response = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a concise expert assistant. Use the internal document context provided to answer the question. "
                    "If the answer is not fully contained in the documents, you may use your own knowledge to supplement it. "
                    "Cite relevant documents (with document name and page number) at the end if used."
                )},
                {"role": "user", "content": f"{context_text}\n\nQuestion: {query}"}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip(), context_chunks
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None, []

# ====== Clear Chat Button ======
if st.button("🗑️ Clear Conversation History"):
    st.session_state.chat_history = []

# ====== User Query Input ======
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
                log_chat_to_excel(query, answer, used_chunks)

# ====== Display Chat History ======
for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
    st.markdown(f"### 🧠 Question {len(st.session_state.chat_history)-i+1}")
    st.markdown(chat["question"])

    st.markdown("### 💡 Answer")
    answer_with_highlight = re.sub(r'\(Source: (.*?), page (\d+)\)', r'📝 **[\1 – p.\2]**', chat["answer"])
    st.markdown(answer_with_highlight, unsafe_allow_html=True)

    st.markdown("### 📚 Sources")
    grouped = defaultdict(list)
    for chunk in chat["sources"]:
        meta = chunk.get("metadata", {})
        doc = meta.get("document", "Unknown")
        page = meta.get("page", "?")
        grouped[doc].append(page)

    for doc, pages in grouped.items():
        page_str = ", ".join(map(str, sorted(set(pages))))
        url = f"{BASE_DOC_URL}{doc.replace(' ', '%20')}"
        st.markdown(f"- [{doc} (pages {page_str})]({url})")

    st.markdown("---")
