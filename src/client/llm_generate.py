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
You are a specialized Q&A assistant for the Tesla Model Y User Manual. Using the information provided in the Context section, answer the following question: "{query}".

### Guidelines
1. **Accuracy**: Your answer must be precise and the sentences should flow naturally.
2. **Format**: Your output MUST strictly follow this format:
   {{Answer}} [{{Citation Number 1}}, {{Citation Number 2}}, ...]
3. **Grounding**: If the answer cannot be found in the provided Context, state "No Answer." Do not hallucinate or add any external information.
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
        max_new_tokens=2048,
        do_sample=True,
    )
    
    new_tokens = output[0][input_ids.shape[-1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    context = """
To Adjust the Shoulder Anchor Height
Model Y is equipped with an adjustable shoulder anchor for each front seat to ensure that the seat belt can be positioned correctly. The seat belt should lie flat across the mid-point of your collar bone while in the correct driving position (see Correct Driving Position). Adjust the height of the shoulder anchor if the seat belt is not positioned correctly:
1. Press and hold the button on the shoulder anchor to release the locking mechanism.
2. While holding the button, move the shoulder anchor up or down, as necessary, to correctly position the seat belt.
3. Release the button on the shoulder anchor so that it locks into position.
"""

    query = "How to adjust the shoulder anchor height?"
    result = request_chat(query, context)
    print(result)