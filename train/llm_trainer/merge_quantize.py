"""
Merge LoRA adapter into base model and quantize to AWQ INT4 in one pass.

The merged float16 model is held in CPU memory only and never saved to disk.
Only the final AWQ INT4 model is written to LLM_QUANTIZED_PATH (~4-5 GB).

Pipeline:
    1. Load base model (CPU, float16)
    2. Load LoRA adapter and merge_and_unload() → full float16 model in memory
    3. Save merged model to a temporary directory
    4. AWQ calibration using full prompts (context + question) from LLM_TRAIN_PATH
    5. AWQ INT4 quantization
    6. Save quantized model to LLM_QUANTIZED_PATH
    7. Remove temporary directory

AWQ config:
    w_bit    : 4
    q_group_size : 128  (standard for LLaMA-family models)
    zero_point   : True
    version      : GEMM  (compatible with vLLM)

Calibration data:
    Questions from LLM_TRAIN_PATH (up to AWQ_CALIB_SAMPLES samples).
    Only the question text is used — no context or answer needed.

Usage:
    python train/llm_trainer/merge_and_quantize.py <path/to/lora_checkpoint>

Example:
    python train/llm_trainer/merge_and_quantize.py data/llm/ckpt/epoch_3
"""

import os
import sys
import json
import logging
import tempfile

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from awq import AutoAWQForCausalLM

from config import (
    LLM_MODEL_PATH,
    LLM_TRAIN_PATH,
    LLM_QUANTIZED_PATH,
)

from src.client.llm_generate_vllm import LLM_CHAT_PROMPT

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AWQ_CALIB_SAMPLES = 256    # number of calibration samples for AWQ
AWQ_CALIB_SEQLEN  = 2048   # sequence length for calibration
AWQ_W_BIT         = 4
AWQ_Q_GROUP_SIZE  = 128

# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def load_calib_data(path: str, n: int) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # use full prompt as calibration input
            prompt = LLM_CHAT_PROMPT.format(
                context = item["context"],
                query   = item["question"],
            )
            texts.append(prompt)
            if len(texts) >= n:
                break
    logger.info(f"Loaded {len(texts)} calibration samples from {path}")
    return texts

# ---------------------------------------------------------------------------
# Step 1 & 2: merge
# ---------------------------------------------------------------------------

def merge_lora(lora_ckpt_path: str, tmp_dir: str) -> None:
    """
    Load base model + LoRA adapter, merge, and save to tmp_dir.
    Runs on CPU to avoid GPU OOM during merge.
    """
    logger.info(f"Loading base model from {LLM_MODEL_PATH} (CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype = torch.float16,
        device_map  = "cpu",
    )

    logger.info(f"Loading LoRA adapter from {lora_ckpt_path}...")
    model = PeftModel.from_pretrained(base, lora_ckpt_path)

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to temporary directory {tmp_dir}...")
    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)

    # free memory
    del model, base
    torch.cuda.empty_cache()
    logger.info("Merge complete, float16 model saved to temp dir.")

# ---------------------------------------------------------------------------
# Step 3: quantize
# ---------------------------------------------------------------------------

def quantize(tmp_dir: str, output_path: str, calib_data: list[str]) -> None:
    """
    Load merged model from tmp_dir, run AWQ INT4 quantization,
    and save to output_path.
    """
    logger.info(f"Loading merged model for AWQ quantization from {tmp_dir}...")
    model = AutoAWQForCausalLM.from_pretrained(
        tmp_dir,
        safetensors = True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tmp_dir)

    quant_config = {
        "zero_point": True,
        "q_group_size": AWQ_Q_GROUP_SIZE,
        "w_bit": AWQ_W_BIT,
        "version": "GEMM",
    }

    logger.info("Running AWQ calibration and quantization...")
    model.quantize(
        tokenizer,
        quant_config = quant_config,
        calib_data   = calib_data,
        max_calib_seq_len = AWQ_CALIB_SEQLEN,
    )

    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Saving quantized model to {output_path}...")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    del model
    torch.cuda.empty_cache()
    logger.info("Quantization complete.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_and_quantize.py <path/to/lora_checkpoint>")
        print("Example: python merge_and_quantize.py data/llm/ckpt/epoch_3")
        sys.exit(1)

    lora_ckpt_path = sys.argv[1]

    if not os.path.exists(lora_ckpt_path):
        logger.error(f"LoRA checkpoint not found: {lora_ckpt_path}")
        sys.exit(1)

    logger.info(f"LoRA checkpoint : {lora_ckpt_path}")
    logger.info(f"Output path     : {LLM_QUANTIZED_PATH}")

    # load calibration data before merge to fail fast if file missing
    calib_data = load_calib_data(LLM_TRAIN_PATH, AWQ_CALIB_SAMPLES)

    # use a temp directory for the merged float16 model
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f"Temporary merge directory: {tmp_dir}")

        # step 1+2: merge LoRA into base model, save to tmp_dir
        merge_lora(lora_ckpt_path, tmp_dir)

        # step 3: quantize from tmp_dir, save to LLM_QUANTIZED_PATH
        quantize(tmp_dir, LLM_QUANTIZED_PATH, calib_data)

    # tmp_dir is automatically deleted here by context manager
    logger.info(f"Temporary directory cleaned up.")
    logger.info(f"Done. Quantized model → {LLM_QUANTIZED_PATH}")


if __name__ == "__main__":
    main()