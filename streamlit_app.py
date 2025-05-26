# streamlit_app.py

import streamlit as st
import numpy as np 
import faiss
import json
import openai
import tiktoken
from collections import defaultdict

# ====== Streamlit Config (MUST be first Streamlit command) ======
st.set_page_config(page_title="ESG Q&A Bot", layout="wide")

# ====== Configuration ======
openai.api_key = st.secrets["openai_api_key"]
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4"
TOP_K = 20
MAX_CONTEXT_TOKENS = 6000

# ====== Load resources ======
@st.cache_resource
def load_resources():
    try:
        index = faiss.read_index("4.my_vector_db.index")
        with open("5.metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

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

# ====== Generate Answer ======
def generate_answer(query, context_chunks):
    encoding = tiktoken.encoding_for_model("gpt-4")
    total_tokens = 0
    context_parts = []

    for chunk in context_chunks:
        chunk_tokens = len(encoding.encode(chunk["text"]))
        if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
            break
        context_parts.append(
            f"- {chunk['text']}\n(Source: {chunk['metadata'].get('document', 'Unknown')} p.{chunk['metadata'].get('page', '?')} | Link: {chunk['metadata'].get('link', 'N/A')})"
        )
        total_tokens += chunk_tokens

    context_text = "\n\n".join(context_parts)

    try:
        response = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always cite document names and links in your response."},
                {"role": "user", "content": f"Use the following sources to answer the question below.\n\n{context_text}\n\nQuestion: {query}"}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip(), context_chunks
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None, []

# ====== Streamlit UI ======
st.title("📘 ESG Document Q&A Assistant")
st.markdown("Ask a question below and get answers backed by internal documents.")

query = st.text_input("🔍 Ask your question:")

if query.strip():
    with st.spinner("Processing..."):
        index, metadata = load_resources()

        if index is None or metadata is None:
            st.stop()

        top_chunks = search_chunks(query, index, metadata)

        if not top_chunks:
            st.warning("No relevant results found.")
        else:
            answer, used_chunks = generate_answer(query, top_chunks)
            if answer:
                st.markdown("### 💡 Answer")
                st.markdown(answer)

                st.markdown("### 📚 Sources")
                grouped = defaultdict(list)
                for chunk in used_chunks:
                    meta = chunk["metadata"]
                    grouped[meta.get("document", "Unknown")].append(meta.get("page", "?"))

                for doc, pages in grouped.items():
                    link = next(
                        (c["metadata"].get("link", "#")
                         for c in used_chunks
                         if c["metadata"].get("document") == doc),
                        "#"
                    )
                    st.markdown(f"- [{doc} (pages {', '.join(map(str, sorted(set(pages))))})]({link})")
