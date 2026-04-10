"""
Split LLM-cleaned docs on <<<SPLIT>>> delimiters, assign global chunk_index,
and upsert all chunks into MongoDB.

Steps:
    1. pickle.load()   — load cleaned docs from CLEAN_DOCS_PATH
    2. split_chunks()  — split each Document on <<<SPLIT>>>, assign zero-based
                         continuous chunk_index and md5(page_content) as unique_id
    3. save()          — validate via ManualInfo schema, upsert into MongoDB
                         collection "manual_text"

Input:
    CLEAN_DOCS_PATH   (config.py) — pickle of List[Document] from parse.py

Output:
    MongoDB "manual_text" collection — documents with fields:
        unique_id    : md5(page_content)
        page_content : str
        metadata     : {source, page, unique_id, chunk_index}
"""

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