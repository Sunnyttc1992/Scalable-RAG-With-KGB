import io

def parse_pdf_bytes(file_bytes: bytes, filename: str):
    """
    Parses a PDF file stream using pypdf.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]

    text_content = "\n\n".join(page.strip() for page in pages if page.strip())
    return text_content, {"filename": filename, "type": "pdf", "page_count": len(pages)}
