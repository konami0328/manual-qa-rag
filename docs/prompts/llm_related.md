# 1. GENERATE (USE LLM TO GENERATE TRAINING DATA AND FINETUNE)
"""
llm_generate.py
Input:  query (str), context (str)
Output: LLM response (str) — answer with citation numbers

Dependencies: openai, python-dotenv
"""

# --- Config (.env) ---
# OPENAI_API_KEY = "..."
# OPENAI_BASE_URL = "..."
# OPENAI_MODEL_NAME = "..."

# --- Prompt ---
# LLM_GENERATE_PROMPT:
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

# --- Function ---
def request_chat(query: str, context: str) -> str:
    # format prompt with query and context
    # call OpenAI API
    # return response string

# 2. CLEAN