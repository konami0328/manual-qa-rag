"""
Pydantic schemas for the FastAPI /ask endpoint.

Request:
    AskRequest  — question string from the user

Response (non-streaming):
    ContextChunk  — a single retrieved chunk with page and rerank score
    AskResponse   — full response with answer, contexts, and retrieval status

For streaming responses, the answer is returned as a sequence of plain text
chunks via Server-Sent Events (SSE). Contexts and retrieval status are sent
as a single JSON object at the end of the stream (see main.py).
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """
    Request body for POST /ask.

    Attributes:
        question: The user's question about the Tesla Model Y.
    """
    question: str = Field(..., min_length=1, description="User question")


class ContextChunk(BaseModel):
    """
    A single retrieved and reranked context chunk.

    Attributes:
        chunk:        Raw text content of the chunk.
        page:         Page number in the Tesla Model Y Owner's Manual.
        rerank_score: Relevance score assigned by the reranker (0.0 – 1.0).
    """
    chunk:        str
    page:         int | str    # str to handle '?' when page metadata is missing
    rerank_score: float


class AskResponse(BaseModel):
    """
    Response body for POST /ask (non-streaming).

    Attributes:
        answer:    Generated answer from the LLM.
        contexts:  Retrieved chunks used as context for generation.
        retrieved: True if at least one chunk passed the reranker threshold,
                   False if no relevant context was found (answer will be
                   the standard refusal string).
    """
    answer:    str
    contexts:  list[ContextChunk]
    retrieved: bool


class Timings(BaseModel):
    retrieve_ms: int
    rerank_ms:   int
    generate_ms: int
    total_ms:    int


class BenchmarkResponse(BaseModel):
    answer:     str
    retrieved:  bool
    num_chunks: int
    timings:    Timings