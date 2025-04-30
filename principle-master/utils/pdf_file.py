import os

import pymupdf


pdf_summary_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "./data/principle-summary.pdf")  # Change this to your actual PDF file path
pdf_full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "./data/principle_full.pdf")  # Change this to your actual PDF file path

def load_principle_book_summary_to_string():
    text = ""
    with pymupdf.open(pdf_summary_path) as pdf:
        for page in pdf:
            text += str(page.get_text().encode("utf8", errors='ignore'))
    return text

def load_principle_book_full_to_string():
    text = ""
    with pymupdf.open(pdf_full_path) as pdf:
        for page in pdf:
            text += str(page.get_text().encode("utf8", errors='ignore'))
    return text

