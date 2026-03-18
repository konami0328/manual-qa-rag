import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.documents import Document

from config import MAX_WORKERS


LLM_CLEAN_PROMPT = """
You are a document formatting assistant for automotive owner's manuals.
Clean the following text extracted from a PDF with a two-column layout:

1. Fix broken line breaks caused by column wrapping: if a line does not end with .!?, merge it with the next line using a space.
2. Remove redundant whitespace and artifacts from PDF extraction.
3. Do NOT rephrase, summarize, reorder, add, or remove any content. Preserve the original wording and structure exactly.

Text to clean:
{}

Cleaned output:
"""

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)


def _clean_doc(doc: Document, max_retries: int = 3) -> Document:
    for attempt in range(max_retries):
        try:
            # create a fresh client per request to avoid connection reuse issues
            c = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["OPENAI_BASE_URL"],
                timeout=60.0,
            )
            response = c.chat.completions.create(
                model=os.environ["OPENAI_MODEL_NAME"],
                messages=[{"role": "user", "content": LLM_CLEAN_PROMPT.format(doc.page_content)}],
                temperature=0,
            )
            return Document(
                page_content=response.choices[0].message.content,
                metadata=doc.metadata,
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def request_llm_clean(docs: List[Document]) -> List[Document]:
    """Clean docs concurrently; only process first 10 for validation."""
    results = [None] * len(docs)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_clean_doc, doc): i for i, doc in enumerate(docs)}
        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            results[i] = future.result()

    return results


if __name__ == "__main__":
    from src.parser.parse import load_pdf
    from config import PDF_FILE

    raw_docs = load_pdf(PDF_FILE)
    cleaned_docs = request_llm_clean(raw_docs)
    print(f"Current model: {os.environ['OPENAI_MODEL_NAME']}")

    for doc in cleaned_docs:
        print(f"\n── page={doc.metadata['page']} ──")
        print(doc.page_content)
        print("=" * 60)