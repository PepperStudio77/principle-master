import os

import pymupdf
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage


def get_local_index_store_dir():
    index_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "./index")
    return index_dir


def load_pdf_from_path(path: str):
    text = ""
    with pymupdf.open(path) as pdf:
        for page in pdf:
            text += str(page.get_text().encode("utf8", errors='ignore'))
    return text


def create_and_persist_index_from_path(path: str):
    local_index_store = get_local_index_store_dir()
    content = load_pdf_from_path(path)
    documents = [Document(text=content)]
    vector_index = VectorStoreIndex.from_documents(documents)
    os.makedirs(local_index_store, exist_ok=True)
    vector_index.storage_context.persist(persist_dir=local_index_store)

def load_persisted_index():
    local_index_store = get_local_index_store_dir()
    storage_context = StorageContext.from_defaults(persist_dir=local_index_store)
    index = load_index_from_storage(storage_context)
    return index