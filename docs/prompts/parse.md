"""
parse.py
Input:  PDF file path (PDF_FILE from config.py)
Output: Chunked Document objects saved to MongoDB (manual_text collection)

Dependencies: pymupdf, langchain, pymongo, python-dotenv
"""

# --- Config (config.py) ---
# PDF_FILE = os.path.join(ROOT, "data", "Owners_Manual.pdf")
# CHUNK_SIZE    = 256                           # tokens
# CHUNK_OVERLAP = 50                            # tokens

# --- Data Model (src/fields/manual_info.py) ---
# class ManualInfo(BaseModel):
#     unique_id:    str
#     page_content: Optional[str]
#     metadata:     dict                        # { source: str, page: int }
#
# unique_id = md5(chunk.page_content)

# --- Pipeline ---

def load_pdf(file_path: str) -> List[Document]:
    # open PDF with fitz
    # for each page: extract text, skip if empty
    # return List[Document] — one Document per page

def chunk(raw_docs: List[Document]) -> List[Document]:
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
    raw_docs = load_pdf(PDF_FILE)
    chunks   = chunk(raw_docs)
    save(chunks)
    print(f"Saved {len(chunks)} chunks to MongoDB")