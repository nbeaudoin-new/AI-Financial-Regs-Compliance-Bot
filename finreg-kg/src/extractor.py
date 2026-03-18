import fitz  # PyMuPDF


def extract_pdf(file_bytes: bytes, filename: str) -> dict:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages.append({"page_num": page_num + 1, "text": text})
    full_text = "\n".join(p["text"] for p in pages)
    doc.close()
    return {"filename": filename, "pages": pages, "full_text": full_text}
