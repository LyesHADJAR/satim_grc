import os
import re
import json
from docx import Document


class SatimPolicyChunker:
    def __init__(self, document_title):
        self.document_title = document_title

    def extract_paragraphs(self, docx_path):
        """Extracts all non-empty paragraphs from the .docx file."""
        document = Document(docx_path)
        return [p.text.strip() for p in document.paragraphs if p.text.strip()]

    def is_all_caps_heading(self, line):
        """Detect ALL CAPS section titles (e.g., COMPLIANCE)."""
        return re.match(r'^[A-Z][A-Z\s\/\-]{2,}$', line.strip())

    def is_numbered_section(self, line):
        """Detect numbered headings like '1.0 Purpose'."""
        return re.match(r'^\d+\.\d+\s+.+', line.strip())

    def chunk_paragraphs(self, paragraphs):
        chunks = []
        current_chunk = {"section_id": "", "section_title": "", "text": ""}

        for line in paragraphs:
            if self.is_all_caps_heading(line):
                if current_chunk["text"]:
                    chunks.append(current_chunk)
                current_chunk = {
                    "section_id": line.strip(),
                    "section_title": line.strip(),
                    "text": ""
                }

            elif self.is_numbered_section(line):
                if current_chunk["text"]:
                    chunks.append(current_chunk)
                parts = line.strip().split(maxsplit=1)
                current_chunk = {
                    "section_id": parts[0],
                    "section_title": parts[1] if len(parts) > 1 else parts[0],
                    "text": ""
                }

            else:
                current_chunk["text"] += line.strip() + "\n"

        if current_chunk["text"]:
            chunks.append(current_chunk)

        return chunks

    def format_chunks(self, chunks):
        formatted = []
        for chunk in chunks:
            formatted.append({
                "text": chunk["text"].strip(),
                "document": self.document_title,
                "section_title": chunk["section_title"]
            })
        return formatted

    def process_docx(self, docx_path):
        paragraphs = self.extract_paragraphs(docx_path)
        chunks = self.chunk_paragraphs(paragraphs)
        return self.format_chunks(chunks)


def process_all_policies(input_folder, output_file):
    all_chunks = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".docx"):
            file_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            policy_title = base_name.replace("-", " ").strip().title()
            print(f"Processing: {filename} → {policy_title}")

            chunker = SatimPolicyChunker(document_title=f"SATIM {policy_title}")
            chunks = chunker.process_docx(file_path)
            all_chunks.extend(chunks)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Total chunks: {len(all_chunks)} written to {output_file}")


if __name__ == "__main__":
    input_folder = "preprocessing\\policies\\satim_policies" 
    output_file = "preprocessing\\policies\\satim_chunks.json"
    process_all_policies(input_folder, output_file)

    # cleaning
    input_file = "preprocessing\\policies\\satim_chunks.json"
    output_file = "preprocessing\\policies\\satim_chunks_cleaned.json"

    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    cleaned_chunks = []
    for chunk in chunks:
        text = chunk["text"]

        # Remove lines of underscores
        text = re.sub(r'_+', '', text)

        # Replace multiple newlines and extra whitespace with single space
        text = re.sub(r'\s*\n\s*', ' ', text)     # Replace newlines with space
        text = re.sub(r'\s{2,}', ' ', text).strip()  # Remove double/triple spaces

        cleaned_chunk = {
            "text": text.strip(),
            "document": chunk["document"],
            "section_title": chunk["section_title"]
        }
        cleaned_chunks.append(cleaned_chunk)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_chunks, f, ensure_ascii=False, indent=2)

    print(f"Final cleaned chunks written to {output_file}")

    
