import re
import json
import unicodedata
from PyPDF2 import PdfReader


class PCIRequirementChunker:
    def __init__(self, document_title: str):
        self.document_title = document_title

    def extract_text_from_pdf(self, pdf_path: str, skip_pages: int = 0) -> str:
        reader = PdfReader(pdf_path)
        all_text = []
        for i, page in enumerate(reader.pages):
            if i < skip_pages:
                continue
            text = page.extract_text()
            if text:
                all_text.append(text)
        return "\n".join(all_text)

    def clean_footer_lines(self, text: str) -> str:
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^Page\s+\d+$', line, re.IGNORECASE):
                continue
            if re.match(r'^Page\s+[ivxlc]+$', line, re.IGNORECASE):
                continue
            if line.startswith("©") or "All Rights Reserved" in line:
                continue
            if "Payment Card Industry Data Security Standard" in line:
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    def clean_text(self, text):
        # Replace known problematic characters
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace('‘', "'").replace('’', "'")
        text = text.replace('–', '-').replace('—', '-').replace('•', ' ')
        text = text.replace('…', '...')
        text = text.replace('\" \"', '')
        text = text.replace('\"', '')
    
        # Remove zero-width and control characters
        text = re.sub(r'[\x00-\x1F\x7F\u200B-\u200D\u2028\u2029\u2060\uFEFF]', '', text)
    
        # Normalize Unicode to NFKD and strip accents/diacritics
        text = unicodedata.normalize("NFKD", text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
    
        # Allow only letters, digits, spaces, and selected punctuation
        allowed_punctuation = r".,;:'\"()\[\]!?*-"
        text = re.sub(rf"[^a-zA-Z0-9 {allowed_punctuation}]", ' ', text)
    
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
    
        return text.strip()

    def chunk_sections(self, full_text: str) -> list:
        print("Chunking by top-level numbered sections (e.g., 1., 2., etc.)...")

        pattern = re.compile(
            r'^\s*(?P<title>\d{1,2}\.\s+[^\n]+)',
            re.MULTILINE
        )

        matches = list(pattern.finditer(full_text))
        chunks = []
        appendix_started = False

        for i, match in enumerate(matches):
            title = self.clean_text(match.group('title').strip())
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            section_text = self.clean_text(full_text[start:end].strip())

            if title.lower().startswith("appendix"):
                appendix_started = True
                appendix_text = self.clean_text(full_text[match.start():].strip())
                break

            if not section_text or not title:
                continue

            chunks.append({
                "section_title": title,
                "document": self.document_title,
                "text": section_text
            })

        if appendix_started:
            chunks.append({
                "section_title": "Appendix",
                "document": self.document_title,
                "text": appendix_text
            })

        return chunks

    def process_pdf(self, pdf_path: str, skip_pages: int = 0) -> list:
        print(f"Extracting text from: {pdf_path}")
        raw_text = self.extract_text_from_pdf(pdf_path, skip_pages=skip_pages)
        cleaned_text = self.clean_footer_lines(raw_text)
        return self.chunk_sections(cleaned_text)

    def save_chunks(self, chunks: list, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"{len(chunks)} sections saved to {output_path}")


if __name__ == "__main__":
    pdf_path = "preprocessing\\norms\\international_norms\\PCI-DSS-v4_0_1.pdf"
    output_json = "preprocessing/norms/international_norms/pci_dss_chunks.json"
    document_title = "PCI-DSS v4.0.1"

    chunker = PCIRequirementChunker(document_title)
    chunks = chunker.process_pdf(pdf_path, skip_pages=3)
    chunker.save_chunks(chunks, output_json)
