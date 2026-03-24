import hashlib
import pickle
import numpy as np
from typing import List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from pymongo.collection import Collection

from config import CLEAN_DOCS_PATH, EMBEDDING_MODEL_PATH, BREAKPOINT_PERCENTILE, MAX_CHUNKS_PER_PAGE
from src.client.mongodb_config import MongoConfig
from src.fields.mongodb_info import ManualInfo


_collection = MongoConfig.get_collection("manual_text")
_model = SentenceTransformer(EMBEDDING_MODEL_PATH)


def semantic_chunk(clean_docs: List[Document]) -> List[Document]:
    """Split pages by semantic similarity between paragraphs."""
    all_chunks = []
    
    for doc in tqdm(clean_docs, desc="Semantic chunking"):
        # Step 1: split by \n\n into paragraphs
        paragraphs = [p.strip() for p in doc.page_content.split("\n\n") if p.strip()]
        
        if len(paragraphs) <= 1:
            # single paragraph, no split needed
            chunk_doc = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
            chunk_doc.metadata["unique_id"] = hashlib.md5(doc.page_content.encode()).hexdigest()
            all_chunks.append(chunk_doc)
            continue
        
        # Step 2: compute embeddings for adjacent paragraphs
        embeddings = _model.encode(paragraphs, show_progress_bar=False)
        
        # Step 3: compute cosine similarity between adjacent pairs
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(sim)
        
        # Step 4: find split points below percentile threshold
        threshold = np.percentile(similarities, 100 - BREAKPOINT_PERCENTILE)
        split_indices = [i+1 for i, sim in enumerate(similarities) if sim < threshold]
        
        # Step 5: create initial chunks
        split_indices = [0] + split_indices + [len(paragraphs)]
        chunks_text = []
        for i in range(len(split_indices) - 1):
            chunk_paras = paragraphs[split_indices[i]:split_indices[i+1]]
            chunks_text.append("\n\n".join(chunk_paras))
        
        # Step 6: enforce MAX_CHUNKS_PER_PAGE by merging most similar neighbors
        while len(chunks_text) > MAX_CHUNKS_PER_PAGE:
            # recompute similarities between adjacent chunks
            chunk_embeddings = _model.encode(chunks_text, show_progress_bar=False)
            chunk_sims = []
            for i in range(len(chunk_embeddings) - 1):
                sim = np.dot(chunk_embeddings[i], chunk_embeddings[i+1]) / (
                    np.linalg.norm(chunk_embeddings[i]) * np.linalg.norm(chunk_embeddings[i+1])
                )
                chunk_sims.append(sim)
            
            # merge the most similar pair
            merge_idx = np.argmax(chunk_sims)
            merged = chunks_text[merge_idx] + "\n\n" + chunks_text[merge_idx + 1]
            chunks_text = chunks_text[:merge_idx] + [merged] + chunks_text[merge_idx+2:]
        
        # Step 7: create Document objects with unique_id
        for chunk_text in chunks_text:
            chunk_doc = Document(page_content=chunk_text, metadata=doc.metadata.copy())
            chunk_doc.metadata["unique_id"] = hashlib.md5(chunk_text.encode()).hexdigest()
            all_chunks.append(chunk_doc)
    
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
    chunks = semantic_chunk(clean_docs)
    save(chunks)
    print(f"Saved {len(chunks)} chunks to MongoDB.")


if __name__ == "__main__":
    main()