import os
import pickle
from typing import List

import fitz
from tqdm import tqdm
from langchain_core.documents import Document

from config import PDF_FILE, CLEAN_DOCS_PATH, PAGE_START, PAGE_END, PAGE_CROP_TOP, PAGE_CROP_BOTTOM
from src.client.llm_clean import request_llm_clean


def load_pdf(file_path: str) -> List[Document]:
    """Extract text from pages [PAGE_START, PAGE_END] inclusive; crop header/footer; skip empty pages."""
    pdf  = fitz.open(file_path)
    docs = []
    for page_num in tqdm(range(PAGE_START - 1, PAGE_END)):
        page = pdf.load_page(page_num)
        clip = fitz.Rect(0, PAGE_CROP_TOP, page.rect.width, page.rect.height - PAGE_CROP_BOTTOM)
        text = page.get_text(clip=clip)
        if not text.strip():
            continue
        docs.append(Document(
            page_content=text,
            metadata={"source": file_path, "page": page_num + 1},
        ))
    return docs


def main():
    raw_docs   = load_pdf(PDF_FILE)
    clean_docs = request_llm_clean(raw_docs)
    pickle.dump(clean_docs, open(CLEAN_DOCS_PATH, "wb"))
    print(f"Saved {len(clean_docs)} cleaned pages → {CLEAN_DOCS_PATH}")


if __name__ == "__main__":
    main()