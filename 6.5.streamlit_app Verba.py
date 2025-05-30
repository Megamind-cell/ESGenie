import streamlit as st
import numpy as np
import faiss
import json
import openai
import tiktoken
from PIL import Image
from collections import defaultdict
import re

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
openai.api_key = st.secrets["openai_api_key"]
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4"
TOP_K = 20
MAX_CONTEXT_TOKENS = 6000

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
        response = openai.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return np.array(response.data[0].embedding, dtype="float32")
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

# ====== Search Function ======
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
        results.append({
            "text": texts[i],
            "score": float(D[0][rank]),
            "metadata": meta_lookup.get(chunk_id, {})
        })
    return results

# ====== Generate GPT Answer ======
def generate_answer(query, context_chunks):
    encoding = tiktoken.encoding_for_model("gpt-4")
    total_tokens = 0
    context_parts = []

    for chunk in context_chunks:
        chunk_tokens = len(encoding.encode(chunk["text"]))
        if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
            break
        context_parts.append(chunk["text"])
        total_tokens += chunk_tokens

    context_text = "\n\n".join(context_parts)

    try:
        response = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise expert assistant. Respond to the question clearly and compactly. Do not cite sources in the response."},
                {"role": "user", "content": f"{context_text}\n\nQuestion: {query}"}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip(), context_chunks
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None, []

# ====== Verbatim Matcher (Regex-based) ======
def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

def extract_matching_sentences(answer, chunks):
    answer_lower = answer.lower()
    grouped = defaultdict(list)
    for chunk in chunks:
        doc = chunk["metadata"].get("document", "Unknown")
        page = chunk["metadata"].get("page", "?")
        for sentence in split_sentences(chunk["text"]):
            sentence_clean = sentence.strip()
            if not sentence_clean or len(sentence_clean.split()) < 6:
                continue
            sentence_lower = sentence_clean.lower()
            word_overlap = sum(1 for word in sentence_lower.split() if word in answer_lower)
            match_ratio = word_overlap / len(sentence_lower.split())
            if match_ratio >= 0.4:
                grouped[doc].append((page, sentence_clean))
    return grouped

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

# ====== Display chat history ======
for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
    st.markdown(f"### 🧠 Question {len(st.session_state.chat_history)-i+1}")
    st.markdown(chat["question"])

    st.markdown("### 💡 Answer")
    st.markdown(chat["answer"])

    verbatim_by_doc = extract_matching_sentences(chat["answer"], chat["sources"])
    if verbatim_by_doc:
        st.markdown("### 📎 Verbatim Source Text")
        for doc, entries in verbatim_by_doc.items():
            st.markdown(f"#### 📄 {doc}")
            for page, sentence in sorted(entries, key=lambda x: x[0]):
                st.markdown(f"**p.{page}** – *{sentence}*")

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
