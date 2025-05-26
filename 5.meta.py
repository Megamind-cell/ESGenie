import json

# Step 1: Load embedded chunks
try:
    with open("3.embedded_chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    print(f"❌ Error loading input JSON: {e}")
    raise

# Step 2: Extract full metadata
try:
    metadata = []
    for item in data:
        entry = {
            "id": item["id"],
            "text": item.get("text", ""),
            "metadata": item.get("metadata", {})
        }
        metadata.append(entry)
except Exception as e:
    print(f"❌ Error preparing metadata: {e}")
    raise

# Step 3: Save cleaned metadata to file
try:
    with open("5.metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("✅ Metadata saved successfully to '5.metadata.json'.")
except Exception as e:
    print(f"❌ Failed to save 5.metadata: {e}")
