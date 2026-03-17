from src.retriever.retrieve_bm25 import BM25Retriever
from src.client.llm_generate import request_chat  # CHANGE LATER
from src.client.mongodb_config import MongoConfig
from langchain_core.documents import Document
from config import TOPK


def infer(query: str, retriever: BM25Retriever) -> str:
    """Retrieve relevant chunks and generate an answer with citations."""
    chunks  = retriever.retrieve_topk(query, topk=TOPK)
    context = "\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(chunks))
    return request_chat(query, context)


def main():
    # load chunks from MongoDB and build BM25 index from pkl if exists
    col      = MongoConfig.get_collection("manual_text")
    docs     = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in col.find()]
    retriever = BM25Retriever(docs)

    print("Tesla Model Y Manual QA — type 'exit' to quit\n")
    while True:
        query = input("Question → ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        print(f"Answer   → {infer(query, retriever)}\n")


if __name__ == "__main__":
    main()