import hashlib
from typing import List, Optional

import fitz
import tiktoken
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pymongo.collection import Collection

from config import PDF_FILE, CHUNK_SIZE, CHUNK_OVERLAP
from src.client.mongodb_config import MongoConfig
from src.fields.mongodb_info import ManualInfo


# --- Data model ---

class ManualInfo(BaseModel):
    unique_id:    str
    page_content: Optional[str]
    metadata:     dict  # { source: str, page: int }


# --- Module-level init ---

_collection: Collection = MongoConfig.get_collection("manual_text")
_encoding   = tiktoken.get_encoding("cl100k_base")
_splitter   = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n"],
    length_function=lambda text: len(_encoding.encode(text)),
)


# --- Pipeline ---

def load_pdf(file_path: str) -> List[Document]:
    """Extract text from each PDF page; skip empty pages."""
    pdf  = fitz.open(file_path)
    docs = []
    for page_num in tqdm(range(len(pdf))):
        text = pdf.load_page(page_num).get_text()
        if not text.strip():
            continue
        docs.append(Document(
            page_content=text,
            metadata={"source": file_path, "page": page_num + 1},
        ))
    return docs


def chunk(raw_docs: List[Document]) -> List[Document]:
    """Split each page into token-bounded chunks; attach unique_id to metadata."""
    chunks = []
    for doc in tqdm(raw_docs):
        for chunk_doc in _splitter.create_documents(
            [doc.page_content], metadatas=[doc.metadata]
        ):
            chunk_doc.metadata["unique_id"] = hashlib.md5(
                chunk_doc.page_content.encode()
            ).hexdigest()
            chunks.append(chunk_doc)
    return chunks


def save(chunks: List[Document]) -> None:
    """Validate each chunk via ManualInfo, then upsert into MongoDB."""
    for doc in chunks:
        try:
            record = ManualInfo(
                unique_id=doc.metadata["unique_id"],
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
        except Exception as e:
            print(f"Validation failed, skipping chunk: {e}")
            continue

        _collection.update_one(
            {"unique_id": record.unique_id},
            {"$set": record.model_dump()},
            upsert=True,
        )


def main():
    _collection.delete_many({})
    print(f"Old data deleted!")

    raw_docs = load_pdf(PDF_FILE)
    chunks   = chunk(raw_docs)
    save(chunks)
    print(f"Saved {len(chunks)} chunks to MongoDB")


if __name__ == "__main__":
    main()