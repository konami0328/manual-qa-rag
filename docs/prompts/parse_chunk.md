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




"""
chunk_index.py
Input:  cleaned List[Document] from pickle (CLEAN_DOCS_PATH from config.py)
Output: chunked Document objects saved to MongoDB (manual_text collection)

Dependencies: langchain, pymongo, tiktoken
"""

# --- Config (config.py) ---
# CLEAN_DOCS_PATH  = os.path.join(ROOT, "data", "clean_docs.pkl")
# CHUNK_SIZE       = 256                        # tokens
# CHUNK_OVERLAP    = 50                         # tokens

# --- Data Model (src/fields/manual_info.py) ---
# class ManualInfo(BaseModel):
#     unique_id:    str
#     page_content: Optional[str]
#     metadata:     dict                        # { source: str, page: int }
#
# unique_id = md5(chunk.page_content)

# --- Pipeline ---

def chunk(clean_docs: List[Document]) -> List[Document]:
    # split each page with RecursiveCharacterTextSplitter(CHUNK_SIZE, CHUNK_OVERLAP)
    # assign unique_id = md5(chunk.page_content) into each chunk's metadata
    # return List[Document] — one Document per chunk

    # TODO:

def save(chunks: List[Document]) -> None:
    # for each chunk:
    #   1. validate via ManualInfo(unique_id, page_content, metadata)
    #   2. upsert into MongoDB by unique_id (update_one, upsert=True)
    #   3. skip if validation fails

def main():
    clean_docs = pickle.load(open(CLEAN_DOCS_PATH, "rb"))
    chunks     = chunk(clean_docs)
    save(chunks)
    print(f"Saved {len(chunks)} chunks to MongoDB")