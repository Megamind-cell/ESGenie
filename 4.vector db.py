import json
import numpy as np
import faiss

with open("3.embedded_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

embeddings = np.array([item["embedding"] for item in data]).astype("float32")
ids = [item["id"] for item in data]
texts = [item.get("text", "") for item in data]

dimension = embeddings.shape[1]
index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
index.add_with_ids(embeddings, np.arange(len(embeddings)))

print("Embeddings stored in 4.Vector DB.")

faiss.write_index(index, "4.my_vector_db.index")
print("4.Vector DB saved to file.")
