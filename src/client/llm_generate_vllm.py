from openai import OpenAI

from config import VLLM_BASE_URL, VLLM_MODEL_NAME

LLM_CHAT_PROMPT = """
### Context
{context}

### Task
You are a Q&A assistant for the Tesla Model Y Owner's Manual.
Answer the question using ONLY the information provided in the Context above.

### Question
{query}

### Output Format
Follow these rules strictly:

1. **Structure**: Be direct and answer only what the question asks.
   Include warnings or safety information from the Context ONLY if they describe 
   a risk that arises from performing the action asked about.
   Use a numbered list ONLY when the answer involves sequential steps that must 
   be followed in order. Otherwise, answer in plain sentences.

2. **Citation**: Where information comes from a specific page, cite the page number 
   at the end of the relevant paragraph or list block, using the format [p.45] or 
   [p.45, p.46]. Use only the page number — do NOT copy the chunk number.
   Do not repeat the same citation on every sentence.
   If no page number is available, omit the citation.

3. **Grounding**: Do not add any information not present in the Context. 
   If the question cannot be answered from the Context, respond only with: 
   "This information is not covered in the provided context."
   Do not describe or summarize what the Context does contain.
"""


_client = OpenAI(
    api_key  = "EMPTY",
    base_url = VLLM_BASE_URL,
)


def request_chat(query: str, context: str) -> str:
    prompt = LLM_CHAT_PROMPT.format(context=context, query=query)
    response = _client.chat.completions.create(
        model    = VLLM_MODEL_NAME,
        messages = [{"role": "user", "content": prompt}],
        max_tokens      = 1500,
        temperature     = 0.3,
    )
    return response.choices[0].message.content

def request_chat_stream(query: str, context: str):
    """Yield LLM output tokens one by one via vLLM streaming."""
    prompt = LLM_CHAT_PROMPT.format(context=context, query=query)
    stream = _client.chat.completions.create(
        model       = VLLM_MODEL_NAME,
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = 1500,
        temperature = 0.3,
        stream      = True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token