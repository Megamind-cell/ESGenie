import openai
import json
import time
import hashlib
import os
import streamlit as st  # ✅ secure secrets usage

# ✅ Use secret key securely (requires secrets.toml or Streamlit Cloud secret)
openai.api_key = st.secrets["openai_api_key"]

# Load chunks
with open("2.token_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load existing embeddings if they exist
existing_embeddings = {}
if os.path.exists("3.embedded_chunks.json"):
    with open("3.embedded_chunks.json", "r", encoding="utf-8") as f:
        for item in json.load(f):
            text_hash = hashlib.md5(item["text"].encode()).hexdigest()
            existing_embeddings[text_hash] = item

# Embedding function
def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = openai.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error: {e}")
        return None

# Generate new embeddings only if not cached
embedded_chunks = []
for chunk in chunks:
    text_hash = hashlib.md5(chunk["text"].encode()).hexdigest()
    if text_hash in existing_embeddings:
        embedded_chunks.append(existing_embeddings[text_hash])
        continue

    embedding = get_embedding(chunk["text"])
    if embedding:
        embedded_chunks.append({
            "id": chunk["id"],
            "embedding": embedding,
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        })
    time.sleep(0.5)  # to avoid rate limiting

# Save
with open("3.embedded_chunks.json", "w", encoding="utf-8") as f:
    json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)

print("✅ Embedding complete.")
