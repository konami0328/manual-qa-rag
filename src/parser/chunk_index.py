import hashlib
import pickle
from typing import List

from tqdm import tqdm
from langchain_core.documents import Document
from pymongo.collection import Collection

from config import CLEAN_DOCS_PATH
from src.client.mongodb_config import MongoConfig
from src.fields.mongodb_info import ManualInfo


_collection: Collection = MongoConfig.get_collection("manual_text")

SPLIT_DELIMITER = "<<<SPLIT>>>"


def split_chunks(clean_docs: List[Document]) -> List[Document]:
    """Split each cleaned page by SPLIT_DELIMITER inserted by LLM."""
    all_chunks = []
    chunk_index = 0                                          # global counter
    for doc in tqdm(clean_docs, desc="Splitting chunks"):
        parts = [p.strip() for p in doc.page_content.split(SPLIT_DELIMITER) if p.strip()]
        for part in parts:
            chunk_doc = Document(
                page_content=part,
                metadata={
                    **doc.metadata,
                    "unique_id":   hashlib.md5(part.encode()).hexdigest(),
                    "chunk_index": chunk_index,              # added
                },
            )
            all_chunks.append(chunk_doc)
            chunk_index += 1                                 # increment per chunk
    return all_chunks


def save(chunks: List[Document]) -> None:
    """Validate and upsert each chunk into MongoDB."""
    for doc in tqdm(chunks, desc="Saving to MongoDB"):
        try:
            record = ManualInfo(
                unique_id=doc.metadata["unique_id"],
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
        except Exception as e:
            print(f"Validation failed, skipping: {e}")
            continue
        _collection.update_one(
            {"unique_id": record.unique_id},
            {"$set": record.model_dump()},
            upsert=True,
        )


def main():
    _collection.delete_many({})
    print("Old data cleared.")
    clean_docs = pickle.load(open(CLEAN_DOCS_PATH, "rb"))
    chunks = split_chunks(clean_docs)
    save(chunks)
    print(f"Saved {len(chunks)} chunks to MongoDB.")


if __name__ == "__main__":
    main()