import fitz  # PyMuPDF
import json
import os
import pandas as pd

def is_probable_table(block):
    """Heuristic to check if a block looks like a table."""
    if "lines" not in block:
        return False
    line_lengths = [len(line["spans"]) for line in block["lines"]]
    return len(line_lengths) >= 2 and min(line_lengths) > 1

def extract_pdf_data(pdf_path, metadata_dict, image_output_dir="extracted_images"):
    doc = fitz.open(pdf_path)
    extracted_data = []
    filename = os.path.basename(pdf_path)
    metadata = metadata_dict.get(filename, {})
    os.makedirs(image_output_dir, exist_ok=True)

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_entry = {
            "document": filename,
            "page": page_num + 1,
            "text_content": "",
            "table_like_content": [],
            "images": [],
            "link": metadata.get("Links", "")
        }

        # Text
        page_entry["text_content"] = page.get_text("text").strip()

        # Tables
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if is_probable_table(block):
                rows = []
                for line in block["lines"]:
                    row = " | ".join([span["text"] for span in line["spans"]])
                    rows.append(row)
                page_entry["table_like_content"].append(rows)

        # Images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            image_name = f"{filename}_page{page_num+1}_img{img_index + 1}.{ext}"
            image_path = os.path.join(image_output_dir, image_name)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            page_entry["images"].append(image_path)

        extracted_data.append(page_entry)

    return extracted_data

# Load metadata
metadata_df = pd.read_excel("210425_Document links.xlsx")
metadata_df.columns = metadata_df.columns.str.strip()
metadata_df.rename(columns={'File Name': 'Document'}, inplace=True)
metadata_dict = metadata_df.set_index('Document').to_dict(orient='index')

# Folder containing PDFs
pdf_folder = r"C:\Users\62184\OneDrive - Bain\Documents\Python Scripts\ESG bot"
all_data = []

for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        all_data.extend(extract_pdf_data(pdf_path, metadata_dict))

# Save output to JSON
with open("1.document_data.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print("✅ Enhanced data with text, tables, and images extracted successfully!")
