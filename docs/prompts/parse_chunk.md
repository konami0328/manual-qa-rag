# 1. PARSE
```
"""
parse.py
Input:  PDF file path (PDF_FILE from config.py)
Output: cleaned List[Document] saved to pickle (CLEAN_DOCS_PATH from config.py)

Dependencies: pymupdf, openai, python-dotenv
"""

# --- Config (config.py) ---
# PDF_FILE         = os.path.join(ROOT, "data", "Owners_Manual.pdf")
# CLEAN_DOCS_PATH  = os.path.join(ROOT, "data", "clean_docs.pkl")
# PAGE_START       = 5                          # first content page (1-indexed, inclusive)
# PAGE_END         = 313                        # last content page (1-indexed, inclusive)
# PAGE_CROP_TOP    = 55                         # pt to crop from top (removes header)
# PAGE_CROP_BOTTOM = 25                         # pt to crop from bottom (removes footer)

# --- Pipeline ---

def load_pdf(file_path: str) -> List[Document]:
    # open PDF with fitz
    # iterate pages in range [PAGE_START, PAGE_END] inclusive
    # for each page: extract text within crop rect (exclude header/footer)
    # skip if empty
    # return List[Document] — one Document per page

def main():
    raw_docs    = load_pdf(PDF_FILE)
    clean_docs  = request_llm_clean(raw_docs)   # from src.client.llm_clean
    pickle.dump(clean_docs, open(CLEAN_DOCS_PATH, "wb"))
    print(f"Saved {len(clean_docs)} cleaned pages to {CLEAN_DOCS_PATH}")
```


# 2. CHUNK
```
"""
chunk_index.py
Input:  cleaned List[Document] from pickle (CLEAN_DOCS_PATH)
Output: chunked Documents saved to MongoDB (manual_text collection)

Dependencies: langchain, pymongo, sentence-transformers
"""

# --- Config (config.py) ---
# CLEAN_DOCS_PATH      = os.path.join(ROOT, "data", "clean_docs.pkl")
# EMBEDDING_MODEL_PATH = "models/BAAI/bge-m3"
# BREAKPOINT_PERCENTILE = 75                    # lower percentile = more splits
# MAX_CHUNKS_PER_PAGE   = 5

# --- Data Model (src/fields/manual_info.py) ---
# class ManualInfo(BaseModel):
#     unique_id:    str                         # md5(page_content)
#     page_content: Optional[str]
#     metadata:     dict                        # {source: str, page: int}

# --- Pipeline ---

def semantic_chunk(clean_docs: List[Document]) -> List[Document]:
    """
    For each page:
    1. Split by \n\n into paragraphs
    2. Compute embedding similarity between adjacent paragraphs (bge-m3)
    3. Split at low-similarity boundaries (below percentile threshold)
    4. If chunks > MAX_CHUNKS_PER_PAGE, merge most similar neighbors
    5. Assign unique_id = md5(page_content) to each chunk
    
    Returns: List[Document] with metadata inherited from original page
    """
    pass

def save(chunks: List[Document]) -> None:
    """
    Validate via ManualInfo, upsert to MongoDB by unique_id
    Skip on validation failure
    """
    pass

def main():
    clean_docs = pickle.load(open(CLEAN_DOCS_PATH, "rb"))
    chunks     = semantic_chunk(clean_docs)
    save(chunks)
    print(f"Saved {len(chunks)} chunks to MongoDB")
```