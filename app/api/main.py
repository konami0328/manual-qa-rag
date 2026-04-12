"""
FastAPI serving for the Tesla Model Y RAG QA system.

Endpoint:
    POST /ask  — streaming SSE response

Stream protocol:
    Each SSE message has a plain-text data field prefixed by a type tag.
    The client splits on the prefix to route content appropriately.

    [CONTEXT] p.<page>  score:<rerank_score>\\n<chunk_text>
        Sent once per retrieved chunk, immediately after retrieval+rerank.
        Gradio renders these in the context panel before the answer starts.

    [TOKEN]<token_text>
        Sent once per LLM output token. Gradio appends to the answer box.

    [DONE]
        Signals end of stream.

    [ERROR]<message>
        Sent if an unrecoverable error occurs.

Usage:
    uvicorn app.api.main:app --host 0.0.0.0 --port 8000
"""

import logging
import time
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document

from app.api.schemas import AskRequest, BenchmarkResponse, Timings
from config import TOPK, GENERATION_TOPK, GENERATION_THRESHOLD
from src.client.mongodb_config import MongoConfig
from src.retriever.retrieve_hybrid import HybridRetriever
from src.reranker.rerank_bge_finetuned import FinetunedReranker
from src.client.llm_generate_vllm import request_chat, request_chat_stream

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Tesla Model Y QA",
    description = "RAG QA system for the Tesla Model Y Owner's Manual",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ---------------------------------------------------------------------------
# Startup: load retriever and reranker once at server start
# ---------------------------------------------------------------------------

_retriever: HybridRetriever | None = None
_reranker:  FinetunedReranker | None = None


@app.on_event("startup")
async def startup():
    global _retriever, _reranker
    logger.info("Loading docs from MongoDB...")
    col  = MongoConfig.get_collection("manual_text")
    docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in col.find()
    ]
    logger.info(f"Docs loaded: {len(docs)}")
    _retriever = HybridRetriever(docs)
    _reranker  = FinetunedReranker()
    logger.info("Retriever and reranker ready.")


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(chunks: list[Document]) -> str:
    return "\n".join(
        f"[Chunk {i+1}, p.{doc.metadata.get('page', '?')}] {doc.page_content}"
        for i, doc in enumerate(chunks)
    )

# ---------------------------------------------------------------------------
# SSE stream generator
# ---------------------------------------------------------------------------

async def _stream(question: str) -> AsyncGenerator[str, None]:
    """
    Full pipeline as an async SSE generator.

    Yields:
        [CONTEXT] messages immediately after retrieval+rerank
        [TOKEN]   messages during LLM generation
        [DONE]    when generation is complete
        [ERROR]   if something goes wrong
    """
    NO_ANSWER = "This information is not covered in the provided context."

    try:
        # --- retrieval + rerank ---
        candidates = _retriever.retrieve(question, topk=TOPK)
        ranked     = _reranker.rerank(question, candidates)
        chunks     = [
            c for c in ranked
            if c.metadata["rerank_score"] >= GENERATION_THRESHOLD
        ][:GENERATION_TOPK]

        if not chunks:
            # no relevant context found — send empty context, then refusal
            yield f"data: [DONE]\n\n"
            return

        # --- send context chunks immediately ---
        for doc in chunks:
            page  = doc.metadata.get("page", "?")
            score = round(float(doc.metadata.get("rerank_score", 0.0)), 4)
            text  = doc.page_content.replace("\n", " ")
            yield f"data: [CONTEXT] p.{page}  score:{score}\n{text}\n\n"

        # --- stream LLM answer ---
        context = _build_context(chunks)
        for token in request_chat_stream(question, context):
            yield f"data: [TOKEN]{token}\n\n"

        yield f"data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: [ERROR]{str(e)}\n\n"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    """
    Stream a RAG answer for the given question.

    Returns a text/event-stream response. See module docstring for
    the full SSE message protocol.
    """
    return StreamingResponse(
        _stream(request.question),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/ask_benchmark")
async def ask_benchmark(request: AskRequest) -> BenchmarkResponse:
    """
    Non-streaming RAG pipeline with per-layer timing.
    Used exclusively for load testing and performance profiling.
    """
    NO_ANSWER = "This information is not covered in the provided context."

    t0 = time.perf_counter()
    candidates = _retriever.retrieve(request.question, topk=TOPK)
    t1 = time.perf_counter()

    ranked = _reranker.rerank(request.question, candidates)
    t2 = time.perf_counter()

    chunks = [
        c for c in ranked
        if c.metadata["rerank_score"] >= GENERATION_THRESHOLD
    ][:GENERATION_TOPK]

    if not chunks:
        t3 = time.perf_counter()
        return BenchmarkResponse(
            answer     = NO_ANSWER,
            retrieved  = False,
            num_chunks = 0,
            timings    = Timings(
                retrieve_ms = round((t1 - t0) * 1000),
                rerank_ms   = round((t2 - t1) * 1000),
                generate_ms = 0,
                total_ms    = round((t3 - t0) * 1000),
            ),
        )

    context = _build_context(chunks)
    answer  = request_chat(request.question, context)
    t3 = time.perf_counter()

    return BenchmarkResponse(
        answer     = answer,
        retrieved  = True,
        num_chunks = len(chunks),
        timings    = Timings(
            retrieve_ms = round((t1 - t0) * 1000),
            rerank_ms   = round((t2 - t1) * 1000),
            generate_ms = round((t3 - t2) * 1000),
            total_ms    = round((t3 - t0) * 1000),
        ),
    )

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}