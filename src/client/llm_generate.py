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
        max_new_tokens=2048,
        do_sample=True,
    )
    
    new_tokens = output[0][input_ids.shape[-1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True)