import io

# Extract text and simple metadata from a PDF file already loaded into memory.
def parse_pdf_bytes(file_bytes: bytes, filename: str):
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]

    text_content = "\n\n".join(page.strip() for page in pages if page.strip())
    return text_content, {"filename": filename, "type": "pdf", "page_count": len(pages)}
