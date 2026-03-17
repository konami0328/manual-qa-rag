import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Client ---
_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)
_model = os.environ["OPENAI_MODEL_NAME"]

# --- Prompt ---
LLM_CHAT_PROMPT = """
### Context
{context}

### Task
You are a specialized Q&A assistant for the Tesla Model Y User Manual. Using the information provided in the Context section, answer the following question: "{query}".

### Guidelines
1. **Accuracy**: Your answer must be precise and the sentences should flow naturally.
2. **Format**: Your output MUST strictly follow this format:
   {{Answer}} [{{Citation Number 1}}, {{Citation Number 2}}, ...]
3. **Grounding**: If the answer cannot be found in the provided Context, state "No Answer." Do not hallucinate or add any external information.
"""


def request_chat(query: str, context: str) -> str:
    """Call OpenAI API and return the response string."""
    prompt = LLM_CHAT_PROMPT.format(context=context, query=query)
    response = _client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    context = """[1] Press and hold the button on the shoulder anchor to release the locking mechanism.
[2] While holding the button, move the shoulder anchor up or down to correctly position the seat belt.
[3] Pull on the seat belt webbing to check that the anchor is locked into position."""

    query  = "How to adjust the shoulder anchor height?"
    result = request_chat(query, context)
    print(result)