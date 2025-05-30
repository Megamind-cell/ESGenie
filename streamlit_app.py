import streamlit as st
import openai
import numpy as np
import pandas as pd
import json
from collections import defaultdict
import faiss
import tiktoken
from PIL import Image

# ====== Config ======
st.set_page_config(page_title="ESGenie – Custom RFP Bot", layout="wide")
openai.api_key = st.secrets["openai_api_key"]
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4"
TOP_K = 20
MAX_CONTEXT_TOKENS = 6000

# ====== Branding ======
logo = Image.open("logo.png")
col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=200)
with col2:
    st.markdown("## **ESGenie**")
    st.markdown("Custom RFP & ESG Document Q&A Assistant")
st.markdown("---")

# ====== Initialize chat history ======
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ====== Load resources ======
@st.cache_resource
def load_resources():
    index = faiss.read_index("4.my_vector_db.index")
    with open("5.metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# ====== Embedding ======
def get_embedding(text):
    try:
        response = openai.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return np.array(response.data[0].embedding, dtype="float32")
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

# ====== Search ======
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
        results.append({
            "text": texts[i],
            "score": float(D[0][rank]),
            "metadata": meta_lookup.get(ids[i], {})
        })
    results.sort(key=lambda x: x["score"])
    return results

# ====== Answer generation ======
def generate_answer(query, context_chunks):
    encoding = tiktoken.encoding_for_model("gpt-4")
    total_tokens = 0
    context_parts = []

    for chunk in context_chunks:
        tokens = len(encoding.encode(chunk["text"]))
        if total_tokens + tokens > MAX_CONTEXT_TOKENS:
            break
        context_parts.append(chunk["text"])
        total_tokens += tokens

    context_text = "\n\n".join(context_parts)

    try:
        response = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Answer clearly and cite sources."},
                {"role": "user", "content": f"{context_text}\n\nQuestion: {query}"}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip(), context_chunks
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None, []

# ====== Save chat to JSON and Excel ======
def log_chat(question, answer, sources):
    # Append to memory
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "sources": sources
    })

    # Export all to file
    data = []
    for chat in st.session_state.chat_history:
        q = chat["question"]
        a = chat["answer"]
        srcs = chat["sources"]
        data.append({
            "Timestamp": pd.Timestamp.now().isoformat(),
            "Question": q,
            "Answer": a,
            "Sources": ", ".join([
                f"{c['metadata'].get('document', 'Unknown')} (page {c['metadata'].get('page', '?')})"
                for c in srcs
            ]),
            "Sources_raw": [
                {
                    "document": c["metadata"].get("document", "Unknown"),
                    "page": c["metadata"].get("page", "?")
                } for c in srcs
            ]
        })

    df = pd.DataFrame(data)
    df[["Timestamp", "Question", "Answer", "Sources"]].to_excel("chat_history_log.xlsx", index=False)
    with open("chat_history_log.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ====== Clear Chat ======
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# ====== Main Interface ======
query = st.text_input("🔍 Ask your question:")
if query.strip():
    with st.spinner("Processing..."):
        index, metadata = load_resources()
        top_chunks = search_chunks(query, index, metadata)

        if not top_chunks:
            st.warning("No relevant results.")
        else:
            answer, used_chunks = generate_answer(query, top_chunks)
            if answer:
                log_chat(query, answer, used_chunks)

# ====== Display History ======
for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
    st.markdown(f"### 🧠 Q{i}: {chat['question']}")
    st.markdown(f"**💡 Answer:** {chat['answer']}")
    st.markdown("**📚 Sources:**")
    grouped = defaultdict(list)
    for chunk in chat["sources"]:
        doc = chunk["metadata"].get("document", "Unknown")
        page = chunk["metadata"].get("page", "?")
        grouped[doc].append(page)
    for doc, pages in grouped.items():
        st.markdown(f"- {doc} (pages {', '.join(map(str, sorted(set(pages))))})")
    st.markdown("---")
