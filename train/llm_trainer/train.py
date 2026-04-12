"""
QLoRA supervised fine-tuning for Llama-3.1-8B-Instruct.

Training objective:
    Standard causal language modeling loss (cross-entropy) computed only
    on assistant (answer) tokens. Prompt tokens are masked to -100.

QLoRA configuration:
    Base model loaded in NF4 4-bit quantization (bitsandbytes).
    LoRA adapter trained in bf16 on top of the frozen quantized base.
    target_modules : q_proj, v_proj
    r              : LLM_LORA_RANK
    alpha          : LLM_LORA_ALPHA
    dropout        : LLM_LORA_DROPOUT

Training setup:
    - 8-bit AdamW optimizer (bitsandbytes) to reduce optimizer state memory
    - Cosine LR schedule with linear warmup (warmup_ratio=0.03)
    - Gradient checkpointing to reduce activation memory
    - Checkpoint saved after every epoch to LLM_CKPT_DIR/epoch_{n}/
      (only LoRA adapter weights are saved, not the quantized base)
    - Training log saved to LLM_CKPT_DIR/train_log.jsonl
"""

import os
import json
import logging

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from bitsandbytes.optim import AdamW8bit
from tqdm import tqdm

from config import (
    LLM_MODEL_PATH,
    LLM_TRAIN_PATH,
    LLM_VAL_PATH,
    LLM_CKPT_DIR,
    LLM_LORA_RANK,
    LLM_LORA_ALPHA,
    LLM_LORA_DROPOUT,
    LLM_LR,
    LLM_BATCH_SIZE,
    LLM_NUM_EPOCHS,
    LLM_MAX_LENGTH,
)
from train.llm_trainer.dataset import SFTDataset, SFTCollator

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

LOG_STEPS = 50

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_qlora(model_path: str) -> AutoModelForCausalLM:
    """
    Load base model in NF4 4-bit quantization for QLoRA training.

    Memory layout:
        - Base model weights : ~5 GB (NF4 quantized, frozen)
        - LoRA adapter       : ~50 MB (bf16, trainable)
        - Activations        : ~1-2 GB (gradient checkpointing enabled)
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,   # nested quantization, saves ~0.4 GB
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config = bnb_config,
        device_map          = {"": 0},   # put on GPU, not allowed offloaded to CPU
    )
    # prepare_model_for_kbit_training:
    #   - casts layer norms to fp32 for training stability
    #   - enables gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def apply_lora(model) -> AutoModelForCausalLM:
    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = LLM_LORA_RANK,
        lora_alpha     = LLM_LORA_ALPHA,
        lora_dropout   = LLM_LORA_DROPOUT,
        target_modules = ["q_proj", "v_proj"],
        bias           = "none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# ---------------------------------------------------------------------------
# Train epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    epoch: int,
    global_step: int,
    log_f,
) -> tuple[float, int]:
    model.train()
    total_loss   = 0.0
    window_loss  = 0.0
    window_steps = 0
    n_batches    = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch} train"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val      = loss.item()
        total_loss   += loss_val
        window_loss  += loss_val
        window_steps += 1
        n_batches    += 1
        global_step  += 1

        if global_step % LOG_STEPS == 0:
            avg_window = round(window_loss / window_steps, 6)
            record = {"epoch": epoch, "step": global_step, "train_loss": avg_window}
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            window_loss  = 0.0
            window_steps = 0

    return total_loss / max(n_batches, 1), global_step

# ---------------------------------------------------------------------------
# Val epoch
# ---------------------------------------------------------------------------

def val_epoch(
    model,
    loader: DataLoader,
    device: str,
    epoch: int,
    global_step: int,
    log_f,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch} val"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,
            )
            total_loss += outputs.loss.item()
            n_batches  += 1

    val_loss = total_loss / max(n_batches, 1)
    record   = {"epoch": epoch, "step": global_step, "val_loss": round(val_loss, 6)}
    log_f.write(json.dumps(record) + "\n")
    log_f.flush()

    return val_loss

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    logger.info(f"Loading tokenizer from {LLM_MODEL_PATH}...")
    tokenizer = load_tokenizer(LLM_MODEL_PATH)

    logger.info(f"Loading model in QLoRA (NF4 4-bit) from {LLM_MODEL_PATH}...")
    model = load_model_qlora(LLM_MODEL_PATH)
    model = apply_lora(model)

    logger.info("Building datasets...")
    train_ds = SFTDataset(LLM_TRAIN_PATH, tokenizer)
    val_ds   = SFTDataset(LLM_VAL_PATH,   tokenizer)
    logger.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    collator     = SFTCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size = LLM_BATCH_SIZE,
        shuffle    = True,
        collate_fn = collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = LLM_BATCH_SIZE,
        shuffle    = False,
        collate_fn = collator,
    )

    # 8-bit AdamW: optimizer states stored in int8 instead of fp32
    # reduces optimizer memory from ~200MB to ~50MB for LoRA params
    optimizer    = AdamW8bit(model.parameters(), lr=LLM_LR, weight_decay=0.01)
    total_steps  = len(train_loader) * LLM_NUM_EPOCHS
    warmup_steps = int(total_steps * 0.03)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )
    logger.info(f"Total steps: {total_steps}  Warmup steps: {warmup_steps}")

    os.makedirs(LLM_CKPT_DIR, exist_ok=True)
    log_path    = os.path.join(LLM_CKPT_DIR, "train_log.jsonl")
    global_step = 0

    with open(log_path, "a") as log_f:
        for epoch in range(1, LLM_NUM_EPOCHS + 1):

            train_loss, global_step = train_epoch(
                model, train_loader, optimizer, scheduler,
                device, epoch, global_step, log_f,
            )

            val_loss = val_epoch(
                model, val_loader,
                device, epoch, global_step, log_f,
            )

            logger.info(
                f"Epoch {epoch}/{LLM_NUM_EPOCHS}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

            # save only LoRA adapter weights (base model is not saved)
            ckpt_path = os.path.join(LLM_CKPT_DIR, f"epoch_{epoch}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            logger.info(f"Checkpoint saved → {ckpt_path}")

    logger.info(f"Training complete. Log → {log_path}")


if __name__ == "__main__":
    main()