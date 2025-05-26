import json
import hashlib
from collections import defaultdict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Load your JSON file
with open("1.document_data.json", "r", encoding="utf-8") as f:
    document_data = json.load(f)

# Combine pages by document, include links
documents = defaultdict(list)
document_links = {}

for entry in document_data:
    documents[entry["document"]].append((entry["page"], entry["content"]))
    document_links[entry["document"]] = entry.get("link", "")

combined_docs = {}
page_mapping = {}
for doc, pages in documents.items():
    pages.sort(key=lambda x: x[0])
    full_text = []
    page_starts = []
    for page_num, content in pages:
        page_starts.append((len("\n\n".join(full_text)), page_num))
        full_text.append(content)
    combined_text = "\n\n".join(full_text)
    combined_docs[doc] = (combined_text, page_starts)

# Token-based chunking using tiktoken (OpenAI tokenizer)
encoding_name = "cl100k_base"
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=encoding_name,
    chunk_size=500,
    chunk_overlap=50
)

# Create chunks
chunks = []
for doc_name, (text, page_starts) in combined_docs.items():
    docs = splitter.create_documents([text])
    link = document_links.get(doc_name, "")
    
    for i, doc in enumerate(docs):
        chunk_intro = doc.page_content[:30].strip().lower()
        chunk_id = hashlib.md5(f"{doc_name}_{i}_{chunk_intro}".encode()).hexdigest()[:12]

        chunk_start = text.find(doc.page_content[:30])
        page_number = next((page for start, page in reversed(page_starts) if chunk_start >= start), 0)

        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": doc.page_content,
            "metadata": {
                "document": doc_name,
                "page": page_number,
                "link": link  # Include the link here
            }
        })

# Save to JSON
with open("2.token_chunks.json", "w", encoding="utf-8") as out:
    json.dump(chunks, out, indent=2, ensure_ascii=False)
