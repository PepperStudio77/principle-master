import tiktoken
import pymupdf


def load_pdf_to_string(file_path):
    text = ""
    with pymupdf.open(file_path) as pdf:
        for page in pdf:
            text += str(page.get_text().encode("utf8", errors='ignore'))
    return text

# Example usage:
pdf_path = "../data/principle-summary.pdf"  # Change this to your actual PDF file path
book_text = load_pdf_to_string(pdf_path)

encoder = tiktoken.encoding_for_model("gpt-4o-mini")

# Load book content from pdf
tokens =encoder.encode(book_text)

print(len(tokens))
