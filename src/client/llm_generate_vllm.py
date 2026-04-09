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

1. **Citation**: Place the source page number in brackets immediately after each 
   sentence or step that uses information from the Context, e.g. [p.45] or [p.45, p.46].

2. **Structure**:
   - If the answer involves sequential steps or actions: use a numbered list, 
     one citation per step.
   - Otherwise: use flowing sentences, one citation per sentence.

3. **Completeness**: Cover all relevant steps and details found in the Context. 
   Do not omit steps.

4. **Grounding**: Do not add any information not present in the Context. 
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
        max_tokens      = 2048,
        temperature     = 0,  # set 0 for eval
    )
    return response.choices[0].message.content