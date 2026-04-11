"""
SFT Dataset for LLM fine-tuning.

Each sample from mine.py is formatted as a chat turn using the model's
apply_chat_template, then tokenized with loss masking applied so that
only the assistant (answer) tokens contribute to the training loss.

Samples that exceed LLM_MAX_LENGTH tokens after tokenization are
discarded rather than truncated to avoid corrupted training signals.

Input format (from mine.py output):
{
    "question":    str,
    "context":     str,
    "answer":      str,
    "sample_type": "positive" | "negative"
}
"""

import json
import logging
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from config import LLM_MAX_LENGTH
from src.client.llm_generate_vllm import LLM_CHAT_PROMPT

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """
    Supervised fine-tuning dataset for Llama-3.1-8B-Instruct.

    Formats each sample as:
        user:      LLM_CHAT_PROMPT filled with (context, question)
        assistant: answer

    Loss is computed only on assistant tokens (prompt tokens masked to -100).
    Samples exceeding LLM_MAX_LENGTH are discarded.
    """

    def __init__(self, path: str, tokenizer: PreTrainedTokenizer):
        """
        Args:
            path:      Path to JSONL file produced by mine.py.
            tokenizer: Tokenizer for Llama-3.1-8B-Instruct. Must have
                       apply_chat_template support.
        """
        self.tokenizer = tokenizer
        self.samples   = []

        raw       = self._load(path)
        discarded = 0

        for item in raw:
            encoded = self._encode(item)
            if encoded is None:
                discarded += 1
                continue
            self.samples.append(encoded)

        logger.info(
            f"Loaded {len(self.samples)} samples from {path} "
            f"(discarded {discarded} over max length {LLM_MAX_LENGTH})"
        )

    # ---------------------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------------------

    def _load(self, path: str) -> list[dict]:
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def _encode(self, item: dict) -> dict | None:
        """
        Tokenize one sample with loss masking on the prompt portion.

        Returns None if the total length exceeds LLM_MAX_LENGTH.
        """
        question = item["question"]
        context  = item["context"]
        answer   = item["answer"]

        # --- build prompt string ---
        prompt = LLM_CHAT_PROMPT.format(context=context, query=question)

        # --- apply chat template separately for prompt and full sequence ---
        # Tokenize prompt only (no answer) to find the boundary for masking
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt = True,
            tokenize              = True,
        )

        # Tokenize answer (without special tokens, will be appended)
        answer_ids = self.tokenizer.encode(
            answer + self.tokenizer.eos_token,
            add_special_tokens = False,
        )

        input_ids = prompt_ids + answer_ids

        # --- discard if too long ---
        if len(input_ids) > LLM_MAX_LENGTH:
            return None

        # --- loss masking: -100 for prompt tokens ---
        labels = [-100] * len(prompt_ids) + answer_ids

        return {
            "input_ids":      input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels":         labels,
        }

    # ---------------------------------------------------------------------------
    # Dataset interface
    # ---------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class SFTCollator:
    """
    Pads input_ids, attention_mask, and labels to the longest sequence
    in the batch. Labels are padded with -100 so padding tokens do not
    contribute to the loss.
    """

    def __init__(self, pad_token_id: int):
        """
        Args:
            pad_token_id: Token ID used for padding input_ids and
                          attention_mask. Typically tokenizer.pad_token_id.
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids      = []
        attention_mask = []
        labels         = []

        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(     x["input_ids"]      + [self.pad_token_id] * pad_len)
            attention_mask.append(x["attention_mask"]  + [0]                 * pad_len)
            labels.append(        x["labels"]          + [-100]              * pad_len)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }