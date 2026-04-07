from src.retriever.retrieve_hybrid import HybridRetriever
from src.reranker.rerank_bge_finetuned import FinetunedReranker
from src.client.llm_generate import request_chat
from src.client.mongodb_config import MongoConfig
from langchain_core.documents import Document
from config import TOPK


def infer(query: str, retriever: HybridRetriever, reranker: FinetunedReranker) -> str:
    """Retrieve → rerank → generate with citations."""
    candidates = retriever.retrieve(query, topk=TOPK)
    chunks     = reranker.rerank(query, candidates)[:TOPK]
    context    = "\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(chunks))
    return request_chat(query, context)


def main():
    col      = MongoConfig.get_collection("manual_text")
    docs     = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in col.find()]
    retriever = HybridRetriever(docs)
    reranker  = FinetunedReranker()

    print("Tesla Model Y Manual QA — type 'exit' to quit\n")
    while True:
        query = input("Question → ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        print(f"Answer   → {infer(query, retriever, reranker)}\n")


if __name__ == "__main__":
    main()