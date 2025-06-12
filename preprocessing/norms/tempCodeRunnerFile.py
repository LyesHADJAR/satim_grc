import re
import json
from PyPDF2 import PdfReader


class PCISectionChunker:
    def __init__(self, document_title: str):
        self.document_title = document_title

    def extract_text_from_pdf(self, pdf_path: str, skip_pages=0) -> str:
        reader = PdfReader(pdf_path)
        all_text = []
        for i, page in enumerate(reader.pages):
            if i < skip_pages:
                continue
            page_text = page.extract_text()
            if page_text:
                all_text.append(page_text)
        return "\n".join(all_text)  # preserve line breaks

    def chunk_by_titles(self, full_text: str):
        # Match section titles like "4.2 Scope of Work" or "2 Introduction"
        pattern = re.compile(r"^\s*(\d+(?:\.\d+)*)(\s+)([A-Z][^\n]{10,})$", re.MULTILINE)
        matches = list(pattern.finditer(full_text))
        chunks = []

        for i, match in enumerate(matches):
            start = match.start()
            full_title_line = match.group(0).strip()
            section_title = match.group(3).strip()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

            section_text = full_text[start:end].strip()
            section_text = section_text[len(full_title_line):].strip()  # remove title line

            cleaned_text = self.clean_text(section_text)

            if len(cleaned_text) > 50:
                chunks.append({
                    "text": cleaned_text,
                    "document": self.document_title,
                    "section_title": section_title
                })

        return chunks

    def clean_text(self, text: str) -> str:
        # Remove footer lines and collapse whitespace
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^(Page\s+\d+|Page\s+[ivxlc]+)$', line, re.IGNORECASE):
                continue
            if line.startswith("©") or "All Rights Reserved" in line:
                continue
            if line.startswith("Payment Card Industry Data Security Standard"):
                continue
            cleaned_lines.append(line)
        cleaned = " ".join(cleaned_lines)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)  # remove double spaces
        return cleaned

    def process_pdf(self, pdf_path: str, skip_pages: int = 0) -> list:
        full_text = self.extract_text_from_pdf(pdf_path, skip_pages=skip_pages)
        return self.chunk_by_titles(full_text)

    def save_chunks(self, chunks: list, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"✅ {len(chunks)} section chunks saved to {output_path}")
