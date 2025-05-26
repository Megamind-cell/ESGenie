import fitz  # PyMuPDF
import json
import os
import pandas as pd

def extract_pdf_data(pdf_path, metadata_dict):
    doc = fitz.open(pdf_path)
    extracted_data = []
    filename = os.path.basename(pdf_path)
    metadata = metadata_dict.get(filename, {})  # get metadata for the file

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        if text.strip():
            data_entry = {
                "document": filename,
                "page": page_num + 1,
                "content": text.strip(),
                "link": metadata.get("Links", "")  # add the document link
            }
            # Add other metadata fields
            extracted_data.append(data_entry)

    return extracted_data

# Load metadata from Excel
metadata_df = pd.read_excel("210425_Document links.xlsx")

# Clean up column names
metadata_df.columns = metadata_df.columns.str.strip()

# Rename for consistency
metadata_df.rename(columns={'File Name': 'Document'}, inplace=True)

# Set 'Document' as the index for mapping
metadata_dict = metadata_df.set_index('Document').to_dict(orient='index')

# Folder containing PDFs
pdf_folder = r"C:\Users\62184\OneDrive - Bain\Documents\Python Scripts\ESG bot"
all_data = []

# Process PDFs
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        all_data.extend(extract_pdf_data(pdf_path, metadata_dict))

# Save output
with open("1.document_data.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=4)

print("Data with metadata and links extracted & saved successfully!")
