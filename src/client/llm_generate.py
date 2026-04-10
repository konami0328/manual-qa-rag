from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- Load model ---
_model_path = "models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
_tokenizer = AutoTokenizer.from_pretrained(_model_path)
_model = AutoModelForCausalLM.from_pretrained(
    _model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
)

# --- Prompt ---
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


def request_chat(query: str, context: str) -> str:
    prompt = LLM_CHAT_PROMPT.format(context=context, query=query)
    messages = [{"role": "user", "content": prompt}]
    
    input_ids = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    output = _model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        pad_token_id=_tokenizer.eos_token_id,
        max_new_tokens=1500,
        do_sample=True,
    )
    
    new_tokens = output[0][input_ids.shape[-1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True)