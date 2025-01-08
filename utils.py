import io

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from a PDF file."""
    # Open the PDF directly from the stream
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")  # Open the PDF from the stream
    
    text = ""
    for page in pdf_document:
        text += page.get_text()
    
    return text
